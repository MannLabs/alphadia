
import logging
import os

from alphadia.extraction.testing import update_datashare

if __name__ == "__main__":



    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting diann psm extraction performance test")


    # Download test filles
    try:
        test_dir = os.environ['TEST_DATA_DIR']
    except KeyError:
        logging.error('TEST_DATA_DIR environtment variable not set')
        raise KeyError from None
    
    logging.info(f"Test data directory: {test_dir}")

    dependencies = {
        'folder_name': '0_brunner_2022_1ng_extraction',
        'file_list': [
            'https://datashare.biochem.mpg.de/s/LypobC5QM9HLl89',
            'https://datashare.biochem.mpg.de/s/H8y7zzQvdEkb42E'
        ]
    }

    output_dir = os.path.join(test_dir, dependencies['folder_name'])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dependency_list = dependencies['file_list']
    for element in dependency_list:
        update_datashare(element, output_dir)