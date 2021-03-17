a,b=None,None

def fun1(x,y):
    global a
    global b
    a=x
    b=y

def get():
    return a,b