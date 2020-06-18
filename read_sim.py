import numpy as np
import h5py
import csv
import matplotlib.pyplot as plt

with h5py.File('simulation.h5', 'r') as h5f:
    length = h5f['length'][0]
    outputs = []
    for i in range(length):
        outputs.append(np.asarray(h5f['data_{}'.format(i)]))

    with open('simulation_csv.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i, output in enumerate(outputs):
            print('======== {} {}'.format(i, output.shape))
            if i <= 8:
                to_print = output[0, :7, :, 0]
            elif i <= 13:
                to_print = np.expand_dims(output[0, :7], axis=0)
            else:
                to_print = output[:]
            
            print(to_print.shape)

            plt.imshow(to_print)
            plt.show()
            
            csv_writer.writerow('')
            csv_writer.writerow('== {} =='.format(i))
            csv_writer.writerows(to_print)


    print('END')