from data_loader import get_data_generators

train_gen, val_gen = get_data_generators("dataset")

print("Train samples:", train_gen.samples)
print("Validation samples:", val_gen.samples)
print("Classes:", train_gen.class_indices)
