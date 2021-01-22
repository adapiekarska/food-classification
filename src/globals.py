TARGET_SIZE_DEFAULT = (64, 64)
TARGET_SIZE_TRANSFER_LEARNING = (128, 128)

TARGET_SIZE = TARGET_SIZE_DEFAULT  # change if needed
IMG_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
CLASS_NUM = 11
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 7
LEARNING_RATE = 0.001