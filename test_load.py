import pickle
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

o = (load_obj('out'))
a = (load_obj('trainout'))
b = (load_obj('testout'))
print(len(o[3]))
print(len(o[4]))
print(len(o[9]))
print(len(o[7]))
print(o)
print(len(a[3]))
print(len(a[4]))
print(len(a[9]))
print(len(a[7]))
print(a)
print(len(b[3]))
print(len(b[4]))
print(len(b[9]))
print(len(b[7]))
print(b)
