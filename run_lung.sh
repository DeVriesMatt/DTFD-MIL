for i in {0..4};
do python olga_new_main.py --mDATA0_dir_train0 "/mnt/nvme0n1/ICCV/lung_tcga/features_train_${i}.pickle" --mDATA0_dir_val0 "/mnt/nvme0n1/ICCV/lung_tcga/features_val_${i}.pickle" --mDATA_dir_test0 "/mnt/nvme0n1/ICCV/lung_tcga/features_test_${i}.pickle" --save_model_path "/mnt/nvme0n1/ICCV/lung_tcga/best_model_${i}.pth" --log_dir "/mnt/nvme0n1/ICCV/lung_tcga/logs_${i}";
done