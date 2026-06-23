# simple_unet.py

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Layer,
    Input, Conv2D, SeparableConv2D, Conv2DTranspose,
    UpSampling2D, MaxPooling2D,
    Dropout, SpatialDropout2D,
    concatenate, GlobalAveragePooling2D,
    Reshape, Dense, multiply, Activation, Add,
    MultiHeadAttention, LayerNormalization as KLayerNorm
)
from tensorflow.keras.models import Model

# --------- Custom GroupNormalization ---------
class GroupNormalization(Layer):
    def __init__(self, groups=8, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups  = groups
        self.axis    = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None or dim % self.groups != 0:
            raise ValueError(f"Channel dimension ({dim}) must be divisible by groups ({self.groups})")
        self.gamma = self.add_weight(name='gamma', shape=(dim,),
                                     initializer='ones', trainable=True)
        self.beta  = self.add_weight(name='beta',  shape=(dim,),
                                     initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        shape = K.int_shape(inputs)
        group_shape = [-1] + list(shape[1:self.axis]) + [self.groups, shape[self.axis]//self.groups]
        x = K.reshape(inputs, group_shape)
        mean = K.mean(x, axis=[1,2,4], keepdims=True)
        var  = K.var(x,  axis=[1,2,4], keepdims=True)
        x = (x - mean) / K.sqrt(var + self.epsilon)
        x = K.reshape(x, K.shape(inputs))
        return x * self.gamma + self.beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'groups': self.groups, 'axis': self.axis, 'epsilon': self.epsilon})
        return cfg

# --- Squeeze-and-Excitation ---
def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1,1,filters))(se)
    se = Conv2D(filters//reduction, (1,1), activation='relu',
                padding='same', kernel_initializer='he_normal')(se)
    se = Conv2D(filters, (1,1), activation='sigmoid',
                padding='same', kernel_initializer='he_normal')(se)
    return multiply([x, se])

# --- Spatial Attention ---
def spatial_attention(x):
    sa = Conv2D(1, (1,1), activation='sigmoid',
                padding='same', kernel_initializer='he_normal')(x)
    return multiply([x, sa])

# --- Attention Gate for U‑Net skips (fixed) ---
def attention_gate(x, g, inter_channels):
    # both x and g have the same spatial dims here
    theta_x = Conv2D(inter_channels, (1,1), padding='same')(x)
    phi_g   = Conv2D(inter_channels, (1,1), padding='same')(g)
    f       = Activation('relu')(Add()([theta_x, phi_g]))
    psi     = Conv2D(1, (1,1), activation='sigmoid', padding='same')(f)
    return multiply([x, psi])

# --- Bottleneck Transformer Block (uses Keras Reshape) ---
def transformer_block(x, num_heads=4, key_dim=64, ff_dim=256, dropout_rate=0.1):
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x_seq = Reshape((H*W, C))(x)
    attn  = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_seq, x_seq)
    attn  = Dropout(dropout_rate)(attn)
    x1    = KLayerNorm()(x_seq + attn)
    ff    = Dense(ff_dim, activation='relu')(x1)
    ff    = Dense(C)(ff)
    ff    = Dropout(dropout_rate)(ff)
    x2    = KLayerNorm()(x1 + ff)
    return Reshape((H, W, C))(x2)

# --- ASPP with static upsampling ---
def atrous_spatial_pyramid_pooling(x):
    H, W = x.shape[1], x.shape[2]
    pool = GlobalAveragePooling2D()(x)
    pool = Reshape((1,1,x.shape[-1]))(pool)
    pool = Conv2D(256, (1,1), activation='relu', padding='same',
                  kernel_initializer='he_normal')(pool)
    pool = UpSampling2D(size=(H, W), interpolation='bilinear')(pool)

    conv_1x1 = Conv2D(256,(1,1),activation='relu',padding='same',
                      kernel_initializer='he_normal')(x)
    conv_r6  = Conv2D(256,(3,3),dilation_rate=6,activation='relu',
                      padding='same',kernel_initializer='he_normal')(x)
    conv_r12 = Conv2D(256,(3,3),dilation_rate=12,activation='relu',
                      padding='same',kernel_initializer='he_normal')(x)
    conv_r18 = Conv2D(256,(3,3),dilation_rate=18,activation='relu',
                      padding='same',kernel_initializer='he_normal')(x)

    concat = concatenate([pool, conv_1x1, conv_r6, conv_r12, conv_r18])
    return Conv2D(512, (1,1), activation='relu', padding='same',
                  kernel_initializer='he_normal')(concat)

# --- Convolutional Block ---
def conv_block(x, filters, drop_rate=0.0):
    shortcut = x
    x = SeparableConv2D(filters,(3,3),padding='same',
                        depthwise_initializer='he_normal',
                        pointwise_initializer='he_normal')(x)
    x = GroupNormalization(groups=max(1,filters//4))(x)
    x = Activation('swish')(x)
    x = SeparableConv2D(filters,(3,3),padding='same',
                        depthwise_initializer='he_normal',
                        pointwise_initializer='he_normal')(x)
    x = GroupNormalization(groups=max(1,filters//4))(x)
    x = Activation('swish')(x)
    x = SpatialDropout2D(drop_rate)(x)
    x = se_block(x)
    x = spatial_attention(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters,(1,1),padding='same',
                          kernel_initializer='he_normal')(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# --- U‑Net Model ---
def enhanced_unet(
    n_classes=13,
    IMG_HEIGHT=496,
    IMG_WIDTH=496,
    IMG_CHANNELS=1
):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Encoder
    c1 = conv_block(inputs,  64, 0.1); p1 = MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1,    128, 0.1); p2 = MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2,    256, 0.2); p3 = MaxPooling2D((2,2))(c3)
    c4 = conv_block(p3,    512, 0.2); p4 = MaxPooling2D((2,2))(c4)

    # Bottleneck
    b = atrous_spatial_pyramid_pooling(p4)
    b = transformer_block(b, num_heads=4, key_dim=64, ff_dim=256, dropout_rate=0.1)
    b = SpatialDropout2D(0.3)(b)

    # Decoder
    u1    = Conv2DTranspose(512,(2,2),strides=(2,2),padding='same',
                             activation='relu',kernel_initializer='he_normal')(b)
    c4_att= attention_gate(c4, u1, inter_channels=256)
    u1    = concatenate([u1, c4_att]); c5 = conv_block(u1,512,0.2)

    u2    = Conv2DTranspose(256,(2,2),strides=(2,2),padding='same',
                             activation='relu',kernel_initializer='he_normal')(c5)
    c3_att= attention_gate(c3, u2, inter_channels=128)
    u2    = concatenate([u2, c3_att]); c6 = conv_block(u2,256,0.2)

    u3    = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same',
                             activation='relu',kernel_initializer='he_normal')(c6)
    c2_att= attention_gate(c2, u3, inter_channels=64)
    u3    = concatenate([u3, c2_att]); c7 = conv_block(u3,128,0.1)

    u4    = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same',
                             activation='relu',kernel_initializer='he_normal')(c7)
    c1_att= attention_gate(c1, u4, inter_channels=32)
    u4    = concatenate([u4, c1_att]); c8 = conv_block(u4,64,0.1)

    outputs = Conv2D(n_classes, (1,1), activation='softmax', dtype='float32')(c8)
    return Model(inputs, outputs)

# --- Losses tuned for thin retinal layer segmentation ---
DEFAULT_CLASS_WEIGHTS = tf.constant([
    0.25,  # BACKGROUND
    0.50,  # VITREUS
    1.00,  # RNFL
    1.05,  # GCL
    1.35,  # IPL
    1.35,  # INL
    1.25,  # OPL
    0.95,  # ONL
    1.30,  # ELM
    1.30,  # EZ
    1.45,  # IZ
    1.20,  # RPE
    0.90,  # CHORIOCAPILLARIS
], dtype=tf.float32)


def classwise_dice_loss(y_true, y_pred, smooth=1e-6, ignore_background=True):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = (0, 1, 2)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice_per_class = (2.0 * intersection + smooth) / (denominator + smooth)

    if ignore_background:
        dice_per_class = dice_per_class[1:]

    return 1.0 - tf.reduce_mean(dice_per_class)


def weighted_categorical_crossentropy(class_weights=DEFAULT_CLASS_WEIGHTS):
    class_weights = tf.cast(class_weights, tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        pixel_weights = tf.reduce_sum(y_true * class_weights, axis=-1)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(pixel_weights * ce)

    return loss


def combined_loss(
    y_true,
    y_pred,
    dice_weight=1.0,
    ce_weight=1.0,
    class_weights=DEFAULT_CLASS_WEIGHTS,
    ignore_background=True,
):
    dice = classwise_dice_loss(
        y_true,
        y_pred,
        ignore_background=ignore_background,
    )
    ce = weighted_categorical_crossentropy(class_weights)(y_true, y_pred)
    return (dice_weight * dice) + (ce_weight * ce)

def get_patch_dataset(X,y,patch_size=256,batch_size=8):
    def crop(i,m):
        return (
            tf.image.random_crop(i,[patch_size,patch_size,1]),
            tf.image.random_crop(m,[patch_size,patch_size,tf.shape(m)[-1]])
        )
    ds = tf.data.Dataset.from_tensor_slices((X,y))
    return ds.map(crop, num_parallel_calls=tf.data.AUTOTUNE)\
             .batch(batch_size).prefetch(tf.data.AUTOTUNE)
