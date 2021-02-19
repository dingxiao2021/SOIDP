%% Main function 20210212 
%Due to the addition of second-order information, the SOIDP algorithm predicts much more interlayer links in actual calculations than the IDP algorithm (this can be verified by experiments), 
%The actual number of interlayer links is fixed. In order to compare IDP algorithm and SOIDP algorithm fairly, we added truncation in the program, 
%The number of interlayer links predicted by the IDP algorithm is recorded under different training set ratios, SOIDP also predicts the same number of interlayer links to stop the iteration. 

filename_lay1='emaillay1.txt';%the first layer of network data 
filename_lay2='emaillay2.txt';%the second layer of network data 
filename_lay1_2_rela='emailrelation.txt'; %Interlayer link data, the first column is the node ID of the first layer, the second column is the node ID of the second layer, and the last ten columns are uniformly distributed random numbers from 1 to 100 to divide the training set. 
known_rate=0.05:0.05:0.5;%different training set ratio 
knownInterval=0.05;%Interval of training set ratio 

%% IDP algorithm
%% The IDP code is shared by the authors of Interlayer link prediction in multiplex social networks: An iterative degree penalty algorithm. 
t1=[];
t2=[];
t3=[];
thisans=Mul_Func_IDP(filename_lay1,filename_lay2,filename_lay1_2_rela,known_rate,knownInterval,10,1);
edges_num=thisans.edges_num;%Store the number of edges predicted for each calculation 
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


%% SOIDP algorithm    
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