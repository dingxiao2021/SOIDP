%% Main function 20210212 
%由于加入了二阶的信息，SOIDP算法在实际运算中预测出的层间链接要比IDP算法多很多（这个可以通过实验验证），
%而实际的层间边数是固定的，为了公平比较IDP算法与SOIDP算法，我们在程序中加了截断，
%每次不同训练集比例下IDP算法预测出多少条层间链接，SOIDP中也预测相同的边数迭代停止。
%Due to the addition of second-order information, the SOIDP algorithm predicts much more inter-layer links in actual calculations than the IDP algorithm (this can be verified by experiments), 
%The actual number of edges between layers is fixed. In order to compare IDP algorithm and SOIDP algorithm fairly, we have added truncation in the program, 
%Each time the IDP algorithm predicts how many interlayer links under different training set ratios, SOIDP also predicts the same number of edges to stop the iteration. 

filename_lay1='emaillay1.txt';%第一层的网络数据 the first layer of network data 
filename_lay2='emaillay2.txt';%第二层的网络数据 the second layer of network data 
filename_lay1_2_rela='emailrelation.txt';%层间边数据，第一列是第一层节点ID，第二列是第二层节点ID，后十列是生成1到100的均匀分布的随机数用以划分训练集。
                                         %Interlayer link data, the first column is the node ID of the first layer, the second column is the node ID of the second layer, and the last ten columns are uniformly distributed random numbers from 1 to 100 to divide the training set. 
known_rate=0.05:0.05:0.5;%不同的训练集比例 different training set ratio 
knownInterval=0.05;%训练集比例的间隔 Interval of training set ratio 

%% IDP算法
t1=[];
t2=[];
t3=[];
thisans=Mul_Func_IDP(filename_lay1,filename_lay2,filename_lay1_2_rela,known_rate,knownInterval,10,1);
edges_num=thisans.edges_num;%存储每次计算预测的边数 Store the number of edges predicted for each calculation 
for theknown_rate=known_rate
    therow=int8(theknown_rate/knownInterval);
    recall(therow,1)=thisans.recall(therow);
    precision(therow,1)=thisans.precision(therow);
    f_measure(therow,1)=thisans.f_measure(therow);
    w1=[recall];
    w2=[precision];
    w3=[f_measure];
end
therecall=mean(recall,2);
theprecision=mean(precision,2);
thef_measure=mean(f_measure,2);
t1=[t1,w1];
t2=[t2,w2];
t3=[t3,w3];


%% SOIDP算法    
h1=[];
h2=[];
h3=[];
for i=[0.01,0.05,0.1]        
    thisans=Mul_Func_SOIDP(filename_lay1,filename_lay2,filename_lay1_2_rela,known_rate,knownInterval,10,1,i,edges_num);
    for theknown_rate=known_rate
        therow=int8(theknown_rate/knownInterval);
        recall(therow,1)=thisans.recall(therow);
        precision(therow,1)=thisans.precision(therow);
        f_measure(therow,1)=thisans.f_measure(therow);
        w1=[recall];
        w2=[precision];
        w3=[f_measure];
    end
therecall=mean(recall,2);
theprecision=mean(precision,2);
thef_measure=mean(f_measure,2);
h1=[h1,w1];
h2=[h2,w2];
h3=[h3,w3];    
end
dlmwrite('e_recall.txt',[t1,h1],'delimiter','\t');
dlmwrite('e_precision.txt',[t2,h2],'delimiter','\t');
dlmwrite('e_F1.txt',[t3,h3],'delimiter','\t');