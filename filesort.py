import os
import shutil

source_folder = "dataset"  # 여기에 소스 폴더 경로 입력
real_folder = "dataset/real"      # 여기에 real 폴더 경로 입력
fake_folder = "dataset/fake"      # 여기에 fake 폴더 경로 입력

os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)


for filename in os.listdir(source_folder):
    if filename.endswith((".jpg", ".png")):
        # 파일 이름이 숫자만으로 이루어진 경우 (00005, 00047 등)
        if filename.split(".")[0].isdigit():
            destination = real_folder
        else:
            destination = fake_folder

        # 파일 이동
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination, filename)
        shutil.move(source_path, destination_path)

print("이미지 정리가 완료되었습니다.")
