import os

class results_folder:
    def verify_create_dir(path):
        try:
            os.stat(path)
        except:
            os.mkdir(path) 	

        if(path[len(path) - 1] != '/'):
            path += '/'

        try:
            os.stat(path + 'log')
        except:
            os.mkdir(path + 'log') 

        try:
            os.stat(path + 'best_expr')
        except:
            os.mkdir(path + 'best_expr')

        return