import os

def visitDir(path):
    if not os.path.isdir(path):
        print('Error: "', path, '" is not a directory or does not exist.')
        return
    else:
	global x
        try:
            for lists in os.listdir(path):
                sub_path = os.path.join(path, lists)
                x += 1
                print('No.', x, ' ', sub_path)
                if os.path.isdir(sub_path):
                    visitDir(sub_path)
        except:
            pass


if __name__ == '__main__':
    x=0
    visitDir('/home/nvidia/workspace/src/detectAndRecog/src/yolo_surface/data/output')
    print('Total Permission Files: ', x)
