import pandas as pd
import matplotlib.pyplot as plt
filename = "E:\GitHub\ML_Project\model\model_comparison_with_errors.csv"
data = pd.read_csv(filename)

err_1 = data["|Error_1|"]
err_2 = data["|Error_2|"]
err_3 = data["|Error_3|"]

mask2 = err_2<20
err_2 = err_2[mask2]
mask3 = err_3<20
err_3 = err_3[mask3]

plt.figure(figsize=(10, 6))
bins = 30 
# plt.hist(err_1, bins=bins, alpha=0.6, label="Error 1", color="blue")
plt.hist(err_2, bins=bins, alpha=0.6, label="Error 2", color="red")
plt.hist(err_3, bins=bins, alpha=0.6, label="Error 3", color="green")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.yscale("log")
plt.title("Histogram of Errors for Two Models")
plt.legend()
plt.grid(True)
plt.show()
