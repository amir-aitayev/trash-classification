from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, img_size=(150, 150), batch_size=32, val_split=0.2):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=val_split,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen
