n = int(2097152)
# M = matrix(RDF, n)

from pathlib import Path
import random
random.seed(int(413))

ans = [random.randint(1, 10) for i in range(n)]
x = [1.0 for i in range(n)]

aelem = [0.0 for i in range(n)]
f = [0.0 for i in range(n)]
aptr = [n]
jptr = []
slides = [-8, -7, -4, -3, 0, 3, 4, 7, 8]

# '''
for i in range(n):
    row_count = 0
    row_sum = 0
    dot = 0.0
    for j in slides:
        if j != 0 and i + j >= 0 and i + j < n:
            a = random.randint(1, 8)
            row_sum += a
            aelem.append(a)
            jptr.append(i + j)
            dot += ans[i + j] * a
            row_count += 1
    aelem[i] = row_sum + 1
    dot += ans[i] * aelem[i]
    f[i] = dot
    aptr.append(aptr[i] + row_count)
# '''

'''
for i in range(n):
    row_count = 0
    row_sum = 0
    dot = 0
    for j in range(-3, 4):
        if j != 0 and i + j >= 0 and i + j < n:
            a = randint(1, 8)
            row_sum += a
            aelem.append(a)
            jptr.append(i + j)
            dot += ans[i + j] * a
            row_count += 1
    aelem[i] = row_sum + 1
    dot += ans[i] * aelem[i]
    f[i] = dot
    aptr.append(aptr[i] + row_count)
'''

'''
density = 40
for i in range(n):
    row_count = 0
    row_sum = 0
    dot = 0
    for j in range(n):
        k = randint(0, 100)
        if k < density:
            a = randint(-5, 5)
            row_sum += abs(a)
            aelem.append(a)
            jptr.append(j)
            dot += ans[j] * a
            row_count += 1
    aelem[i] = row_sum + 1
    dot += ans[i] * aelem[i]
    f[i] = dot
    aptr.append(aptr[i] + row_count)
'''

Path('slae').mkdir(exist_ok=True)

with open('slae/mat', 'w', encoding="utf-8") as f_mat:
    f_mat.write(f"{len(aelem)}\n")
    for a in aelem:
        f_mat.write(f"{a}\n")

with open('slae/f', 'w', encoding="utf-8") as f_f:
    f_f.write(f"{n}\n")
    for a in f:
        f_f.write(f"{a}\n")

with open('slae/aptr', 'w', encoding="utf-8") as f_aptr:
    f_aptr.write(f"{len(aptr)}\n")
    for a in aptr:
        f_aptr.write(f"{a}\n")

with open('slae/jptr', 'w', encoding="utf-8") as f_jptr:
    f_jptr.write(f"{len(jptr)}\n")
    for a in jptr:
        f_jptr.write(f"{a}\n")

with open('slae/x', 'w', encoding="utf-8") as f_x:
    f_x.write(f"{n}\n")
    for a in x:
        f_x.write(f"{a}\n")

with open('slae/ans', 'w', encoding="utf-8") as f_ans:
    f_ans.write(f"{n}\n")
    for a in ans:
        f_ans.write(f"{a}\n")

print("exit")
'''
r
r = z = f - M*x
p = M*r

rr = r*r
pp = p*p
it = 0
while pp > 1e-6:
    pp = p*p
    alpha = (p*r)/(pp)
    # print (z)
    x += alpha*z
    r -= alpha*p
    # print (x)
    
    ar = M*r
    beta = -(p*ar)/(pp)
    z = r + (z*beta)
    p = ar + (p*beta)

    rr -= alpha*alpha*pp
    it += 1

print ("it = ", it)
print (x)
'''
