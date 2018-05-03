import numpy as np

a = np.array([1, 2, 3])
print(type(a))

print( a.shape )

print( a[0], a[1], a[2] )

a[0] = 5
print(a)

b = np.array( [  [1,2,3], [4,5,6] ] )
print( b.shape )

print( b[0,0], b[0,1], b[1,0] )


a = np.zeros( (2,2) )
print(a)

b = np.ones( (1, 2) )
print(b)

c = np.full( (2,2), 7 )
print( c )

d = np.eye( 2 )
print(d)

e = np.random.random( (2,2) )
print(e)


a = np.array( [ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12] ] )
print(a)

b = a[:2, 1:3]
print(b)


print( a[0, 1] )

b[0,0] = 77
print(a[0,1])


a[0,1] = 2
print(a)


row_r1 = a[1, :]
row_r2 = a[1:2, :]
print(  row_r1, row_r1.shape )
print( row_r2, row_r2.shape )



col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print( col_r1, col_r1.shape )
print( col_r2, col_r2.shape )



a = np.array( [ [1, 2], [3, 4], [5, 6] ] )


print( a[ [0, 1, 2], [0, 1, 0] ] )
print( [ a[0,0], a[1,1], a[2,0] ] )



print( np.array( a[ [0, 1, 2], [0, 1, 0] ] ) )
print( np.array( [ a[0,0], a[1,1], a[2,0] ] ) )


a = np.array( [ [1,2,3], [4,5,6], [7,8,9], [10,11,12] ] )
print(a)
b = np.array( [0, 2, 0, 1] )
print(b)


print( a[ np.arange(4), b ] )
print(a)
a[ np.arange(4), b ] += 10
print(a)

a = np.array( [ [1,2], [3,4], [5,6] ] )
bool_idx = (a>2)
print( bool_idx )


print( a[bool_idx] )
print( a[a>2] )




# Data types
x = np.array([1, 2])
print(x.dtype)

x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([1.1, 2.2], dtype = np.int64)
print(x)



x = np.array( [ [1, 2], [3, 4] ] )
y = np.array( [ [5, 6], [7, 8] ] )
print(x + y)
print( np.add(x, y) )

print(  1/ x)

print( np.dot(x, y) )
print( x.dot(y) )

print( np.sum(x, axis = 0) )
print( np.max(x, axis = 0) )

print( np.sum(x, axis = 1) )
print( np.max(x, axis = 1) )


x = np.array( [ [1, 2, 3, 4] ] )
print( x )
print( x.T )


v = np.array( [1, 2, 3] )
print( v )
print( v.T )

#Broadcast
x = np.array( [ [1,2,3], [4,5,6], [7,8,9], [10,11,12] ])
v = np.array( [ [1,2,3,4] ])
v = v.T
y = x + v;
print(y)
