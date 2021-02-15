function [ net ] = FormNet( linklist )
%% 该函数用于邻接矩阵的稀疏存储，输出为三元组表示 
%% This function is used for sparse storage of the adjacency matrix, and the output is represented by triples.
    if ~all(all(linklist(:,1:2)))
        linklist(:,1:2) = linklist(:,1:2)+1;
    end
    linklist(:,3) = 1;
    net = spconvert(linklist);
    nodenum = length(net);
    net(nodenum,nodenum) = 0;                               
    net = net-diag(diag(net));
    net = spones(net + net'); 
end 