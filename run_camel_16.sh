for i in 4;
do python Main_DTFD_MIL.py --task 'evaluate' --mDATA0_dir_train0 "/mnt/nvme0n1/ICCV/camelyon_5_fold/v2/features_256_train_${i}.pickle" --mDATA0_dir_val0 "/mnt/nvme0n1/ICCV/camelyon_5_fold/v2/features_256_val_${i}.pickle" --mDATA_dir_test0 "/mnt/nvme0n1/ICCV/camelyon_5_fold/v2/features_256_test_${i}.pickle" --saved_model_path "/mnt/nvme0n1/ICCV/camelyon_5_fold/v2/best_model_${i}.pth";
done