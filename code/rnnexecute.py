# script2.py
import subprocess

# Call script1.py with arguments
l1=['25','50']
l2=['0','2','5']
l3=['0.5','0.1','0.05']
import itertools
# for i in itertools.product(l1,l2,l3):
#     print("#######################################################")
#     print(i)
#     subprocess.run(["python", r"C:\Users\Vaikunth Guruswamy\Downloads\nlu\code\runner.py", 'train-lm-rnn', '.\data\\',i[0],i[1],i[2]])

l1=['10','25','50']
l2=['0']
l3=['0.5']
for i in itertools.product(l1,l2,l3):
    print("#######################################################")
    print("<------------RNN------------->")
    print(i)
    subprocess.run(["python", r"C:\Users\Vaikunth Guruswamy\Downloads\nlu\code\runner.py", 'train-np-rnn', '.\data\\',i[0],i[1],i[2]])
for i in itertools.product(l1,l2,l3):
    print("#######################################################")
    print("<------------GRU------------->")
    print(i)
    subprocess.run(["python", r"C:\Users\Vaikunth Guruswamy\Downloads\nlu\code\runner.py", 'train-np-GRU', '.\data\\',i[0],i[1],i[2]])
