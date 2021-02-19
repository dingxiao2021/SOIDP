function [ net ] = FormNet( linklist )
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