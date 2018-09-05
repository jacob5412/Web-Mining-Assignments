from math import log,ceil

log2 = lambda x: log(x,2)

def binary(x, l = 1):
	fmt = '{0:0%db}'%1
	return fmt.format(x)

def unary(x):
	return x*'1'+'0'

def elias_generic(lencoding, x):
	if x == 0: return '0'
	l = 1+int(log2(x))
	a = x - 2**(int(log2(x)))
	k = int(log2(x))
	return lencoding(l) + binary(a,k)

def golomb(b, x):
	q = int((x) / b)
	r = int((x) % b)
	l = int(ceil(log2(b)))
	#print(q,r,l)
	return unary(q) + binary(r, l)

def elias_gamma(x):
    return elias_generic(unary, x)

def elias_delta(x):
    return elias_generic(elias_gamma,x)

print("%-46s %-20s" %
      (" ","Elias"))
print("%5s: %-11s : %-10s : %-10s : %-10s : %-10s" %
      ("Num", "Unary", "Binary", "Gamma", "Delta", "Goloumb"))
for i in range(11):
	print("%5d: %-11s : %-10s : %-10s : %-10s : %-10s" %
	(i,unary(i),binary(i),elias_gamma(i),elias_delta(i), golomb(3,i)))
