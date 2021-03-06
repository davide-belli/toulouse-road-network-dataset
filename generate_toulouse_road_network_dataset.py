from utils.generate_bins import main_generate_bins
from utils.generate_datapoints import main_generate_datapoints
from utils.generate_image_arrays import main_generate_image_arrays

if __name__ == '__main__':
    main_generate_bins()
    main_generate_datapoints()
    main_generate_image_arrays()
    print("Dataset generation completed")