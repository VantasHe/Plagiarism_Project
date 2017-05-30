import os

def get_index_dir(dm_path):
    file_index = []
    for file in os.listdir(dm_path):
        file_index.append(file)
        #file_index.append(os.path.join(dm_path, file))
    return file_index

def get_walkthrought_dir(dm_path):
    """     return 3 parameter:
                file_index[0]: total path infomation
                file_index[1]: file path directory
                file_index[2]: file name
    """
    file_index = []
    for dirPath, dirName, fileName in os.walk(dm_path):
        for file in fileName:
            path_info = [os.path.join(dirPath, file), dirPath, file]
            file_index.append(path_info)
    return file_index

if __name__ == "__main__":
    print("======Test get index in directory======")
    dir_path = "/Users/vick/Documents/Python/Training_Data/Html_CityU2"
    print(get_index_dir(dm_path=dir_path))
