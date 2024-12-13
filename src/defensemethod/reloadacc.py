import torch

# 저장된 경로에서 체크포인트 불러오기
checkpoint_path = r"C:\Users\user\Desktop\Adversarial-Training-and-Spatial-Smoothing-master\Adversarial-Training-and-Spatial-Smoothing-master\src\model\metrics\_hagishiro_adv_train_Checkpoint.pth"
checkpoint = torch.load(checkpoint_path)

# 저장된 손실값과 정확도 가져오기
loss_per_epoch = checkpoint.get("Loss")  # 손실값 리스트
acc_per_epoch = checkpoint.get("Acc")    # 정확도 리스트

# 출력
print("Loss per epoch:", loss_per_epoch)
print("Accuracy per epoch:", acc_per_epoch)
