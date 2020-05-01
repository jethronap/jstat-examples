"""
Utility script for plotting with matplotlib
"""
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def main():

  sns.set(color_codes=True)
  files = ['averages10.csv', 'averages100.csv', 'averages500.csv', 'averages999.csv']

  for file in files:
    with open(file, 'r', newline='') as csvfile:
      csv_file_reader = csv.reader(csvfile, delimiter=",")
      data=[]
      for row in csv_file_reader:
        #row_vals = row.split(",")
        data = [ float(val) for val in row]


      sns.distplot(data, bins=25, kde=False, rug=True)
      plt.xlabel('Means')
      plt.show()


if __name__ == '__main__':
	main()
	#main("knn_classifier_3.csv")



