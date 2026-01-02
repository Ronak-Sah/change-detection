from src.components.inference import Prediction
import matplotlib.pyplot as plt
before_path=r"D:\Ml Dl\Project\change-detection\artifacts\data_ingestion\LEVIR CD\test\B\test_126.png"
after_path=r"D:\Ml Dl\Project\change-detection\artifacts\data_ingestion\LEVIR CD\test\A\test_126.png"


prediction=Prediction(before_path,after_path)
result=prediction.predict()


plt.figure(figsize=(6,6))
plt.imshow(result, cmap="gray")
plt.title("Change Detection Mask")
plt.axis("off")
plt.show()
