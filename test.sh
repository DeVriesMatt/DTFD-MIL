for i in {0..4};
do python Main_DTFD_MIL.py --mDATA0_dir_train0 "/mnt/nvme0n1/ICCV/lipo/splits/features_256_train_${i}.pickle" --mDATA0_dir_val0 "/mnt/nvme0n1/ICCV/lipo/splits/features_256_val_${i}.pickle" --mDATA_dir_test0 "/mnt/nvme0n1/ICCV/lipo/splits/features_256_test_${i}.pickle" --saved_model_path "/mnt/nvme0n1/ICCV/lipo/splits/best_model_${i}.pth";
done