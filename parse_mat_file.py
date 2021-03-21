import sys
import scipy.io as sio


def main():
    filename = sys.argv[1]
    mat_contents = sio.loadmat(filename)
    print(mat_contents)

if __name__ == '__main__':
    main()
