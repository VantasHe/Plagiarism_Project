import os
import sys
import string


def lookupFileList(path):
    """Lookup the directory index and return each file."""

    fileList = []
    files = os.listdir(path)
    for f in files:
        if(f[0]) == '.':
            pass
        else :
            fileList.append(f)
    return fileList


def lookupAllFiles(rootPath):
    """Lookup the directory index and return full path of files."""

    fileList = []
    for root, dirs, files in os.walk(rootPath) :
        for name in dirs:
            if name[0] == '.':
                dirs.remove(name)
        for name in files:
            if name[0] == '.':
                pass
            else :
                fileList.append(root + '\\' + name)
    return fileList

if __name__ == '__main__':
    path = 'D:\\dataset\\pan09-external-plagiarism-detection-test-corpus-2009-05-21\\source-documents'
    print(lookupAllFiles(path))