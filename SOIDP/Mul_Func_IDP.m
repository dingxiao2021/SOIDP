%% The IDP code is shared by the authors of Interlayer link prediction in multiplex social networks: An iterative degree penalty algorithm. 
function [output_args] = Mul_Func_IDP(filename_lay1,filename_lay2,filename_lay1_2_rela,theknown_rate,krIntval,repetitiontimes,thre)%krIntval is the real interval rate
%function [output_args] = Mul_Func_IDP(filename_lay1,filename_lay2,filename_lay1_2_rela,theknown_rate,krIntval,repetitiontimes,thre,eps)%krIntval is the real interval rate
% SOIDP 20210212
%filename_lay1,filename_lay2,filename_lay1_2_rela:Two-layer network data and interlayer data
%theknown_rate:Training set ratio (0.05:0.05:0.5)
%krIntval:Training set interval (0.05)
%repetitiontimes:Average times (10)
%thre:delta(1)
lay1_linklist=textread(filename_lay1);
lay2_linklist=textread(filename_lay2);
lay1_2_relation=textread(filename_lay1_2_rela);
lay1_linklist=FormNet(lay1_linklist); %The first-order adjacency matrix of the first layer network 
lay2_linklist=FormNet(lay2_linklist); %The first-order adjacency matrix of the second layer network
% lay1_linklist_2=lay1_linklist^2-diag(diag(lay1_linklist^2)); 
% lay1_linklist_2(lay1_linklist_2~=0)=1;
% lay1_linklist_2=lay1_linklist_2-lay1_linklist;
% lay1_linklist_2(lay1_linklist_2==-1)=0;
% lay2_linklist_2=lay2_linklist^2-diag(diag(lay2_linklist^2));
% lay2_linklist_2(lay2_linklist_2~=0)=1;
% lay2_linklist_2=lay2_linklist_2-lay2_linklist;
% lay2_linklist_2(lay2_linklist_2==-1)=0; 
lay1_node_num=size(lay1_linklist,1);  %Number of first layer network nodes 
lay2_node_num=size(lay2_linklist,2);  %Number of second layer network nodes 

%Calculate Degree of all nodes
%Node degree: the first column is the node number, the second column is the node degree 
lay1_degree(:,1)=1:size(lay1_linklist,1);
lay2_degree(:,1)=1:size(lay2_linklist,1);
lay1_degree(:,2)=full(sum(lay1_linklist,2));
lay2_degree(:,2)=full(sum(lay2_linklist,2));

%Processing the extracted network has a node degree of 0 
%Delete the case where the degree of the known inter-layer node pair (training set) is 0 
lay1_2_relation1=lay1_2_relation;
lay1_2_relation1(:,3)=lay1_degree(lay1_2_relation(:,1),2);
lay1_2_relation1(:,4)=lay2_degree(lay1_2_relation(:,2),2);
lay1_2_relation1(:,5)=lay1_2_relation1(:,3).*lay1_2_relation1(:,4);
row=find(lay1_2_relation1(:,5)==0);
lay1_2_relation(row,:)=[];
clear lay1_2_relation1;
lay1_2_relation1=lay1_2_relation;

%Perform smoothing processing to add one to the degree of each layer node to prevent the denominator from being 0 in the matching degree formula 
lay1_degree(:,2)=lay1_degree(:,2)+1;
lay2_degree(:,2)=lay2_degree(:,2)+1;

%Repeat 10 times to label train and probe sets£¬storing recall and precision 
recall=zeros(theknown_rate(1,size(theknown_rate,2))/krIntval,repetitiontimes);
precision=zeros(theknown_rate(1,size(theknown_rate,2))/krIntval,repetitiontimes); 
edges_num=zeros(theknown_rate(1,size(theknown_rate,2))/krIntval,repetitiontimes);
for experi_times=1:1:repetitiontimes  %Train 10 times 
    %Divide training set and test set
    lay1_2_relation=lay1_2_relation(:,1:2);
    lay1_2_relation(:,3)=lay1_2_relation1(:,experi_times+2);  %3-13 columns as labels, train 10 times, select one column each time
    
    for known_rate=theknown_rate  % p:0.05:0.05:0.5
        for testlabel=known_rate*100 
            kno_num=0;  %Number of training sets
            unkno_num=0; %Number of test sets 
            lay1_2_rela_kno=zeros(lay1_node_num,2); %Training sets
            lay1_2_rela_unkno=zeros(lay2_node_num,2); %Test sets
            for i=1:size(lay1_2_relation)
                if lay1_2_relation(i,3)<=testlabel  %If the label of the node pair between the layers <= the divided label value
                   kno_num=kno_num+1;                %Node pairs are divided into training set 
                   lay1_2_rela_kno(kno_num,1)=lay1_2_relation(i,1);
                   lay1_2_rela_kno(kno_num,2)=lay1_2_relation(i,2);
                else                                
                   unkno_num=unkno_num+1;           %Node pairs are divided into test set
                   lay1_2_rela_unkno(unkno_num,1)=lay1_2_relation(i,1);
                   lay1_2_rela_unkno(unkno_num,2)=lay1_2_relation(i,2);
                end
            end
            lay1_2_rela_kno=lay1_2_rela_kno(1:kno_num,1:2); %Training set: The first column: node ID in layer 1; The second column: node ID in layer 2
            lay1_2_rela_unkno=lay1_2_rela_unkno(1:unkno_num,1:2); %Test set
            lay1_kno_num=0; %Number of matched nodes in layer 1 
            lay2_kno_num=0; %Number of matched nodes in layer 2            
            %kno new old id
            lay1_2_rela_kno(:,3)=1:kno_num;
            lay1_kno_new_oldID=lay1_2_rela_kno(:,[3,1]); %The first column: sequence number 1, 2, 3...; The second column: matched node id 
            lay2_kno_new_oldID=lay1_2_rela_kno(:,[3,2]); 

            %Renumber the unmatched nodes in order, for example, the original ID is 1, 2, 4, and the new ID is 1, 2, 3 
            lay1_unkno_new_oldID(:,1)=1:(size(lay1_linklist,1)-kno_num);
            lay1_unkno_new_oldID(:,2)=setdiff(lay1_degree(:,1),lay1_kno_new_oldID(:,2)); 
            lay2_unkno_new_oldID(:,1)=1:(size(lay2_linklist,1)-kno_num);
            lay2_unkno_new_oldID(:,2)=setdiff(lay2_degree(:,1),lay2_kno_new_oldID(:,2)); 
            lay1_unkno_new_oldID(:,3)=lay1_degree(lay1_unkno_new_oldID(:,2),2);  %The first column: the new ID; the second column: the original ID; the third column: the degree of the node 
            lay2_unkno_new_oldID(:,3)=lay2_degree(lay2_unkno_new_oldID(:,2),2);

            lay2_kno_new_oldID_sortnew=sortrows(lay2_kno_new_oldID,1);

            %Generate two intermediate matrices E1, E2 (sub-matrix of adjacency matrix A) 
            lay1_unkno_kno_spmat_list=lay1_linklist(lay1_unkno_new_oldID(:,2),lay1_kno_new_oldID(:,2)); 
            lay2_kno_unkno_spmat_list=lay2_linklist(lay2_kno_new_oldID_sortnew(:,2),lay2_unkno_new_oldID(:,2));
            
%             Generate second-order intermediate matrix U1, U2 (sub-matrix of second-order adjacency matrix B) 
%             lay1_unkno_kno_spmat_list_2=lay1_linklist_2(lay1_unkno_new_oldID(:,2),lay1_kno_new_oldID(:,2)); 
%             lay2_kno_unkno_spmat_list_2=lay2_linklist_2(lay2_kno_new_oldID_sortnew(:,2),lay2_unkno_new_oldID(:,2));

            %Matrix dot product(H1)'.*(E1)'; (E2).*(H2)
            lay1_unkno_kno_spmat_list1=lay1_unkno_kno_spmat_list.*(repmat(1./(log(lay1_degree(lay1_kno_new_oldID(:,2),2)))',[size(lay1_unkno_new_oldID,1),1]));
            lay1_unkno_kno_spmat_list1(isnan(lay1_unkno_kno_spmat_list1))=0;
            lay1_unkno_kno_spmat_list1(isinf(lay1_unkno_kno_spmat_list1))=0;
            lay2_kno_unkno_spmat_list1=lay2_kno_unkno_spmat_list.*(repmat(1./(log(lay2_degree(lay2_kno_new_oldID_sortnew(:,2),2))),[1,size(lay2_unkno_new_oldID,1)]));
            lay2_kno_unkno_spmat_list1(isnan(lay2_kno_unkno_spmat_list1))=0;
            lay2_kno_unkno_spmat_list1(isinf(lay2_kno_unkno_spmat_list1))=0;
            
%             %Matrix dot product (L1)'.*(U1)' ; (U2).*(L2)
%             lay1_unkno_kno_spmat_list1_2=lay1_unkno_kno_spmat_list_2.*(repmat(1./(log(lay1_degree(lay1_kno_new_oldID(:,2),2)))',[size(lay1_unkno_new_oldID,1),1]));
%             lay1_unkno_kno_spmat_list1_2(isnan(lay1_unkno_kno_spmat_list1_2))=0;
%             lay1_unkno_kno_spmat_list1_2(isinf(lay1_unkno_kno_spmat_list1_2))=0;
%             lay2_kno_unkno_spmat_list1_2=lay2_kno_unkno_spmat_list_2.*(repmat(1./(log(lay2_degree(lay2_kno_new_oldID_sortnew(:,2),2))),[1,size(lay2_unkno_new_oldID,1)]));
%             lay2_kno_unkno_spmat_list1_2(isnan(lay2_kno_unkno_spmat_list1_2))=0;
%             lay2_kno_unkno_spmat_list1_2(isinf(lay2_kno_unkno_spmat_list1_2))=0;
            
            %Matrix product
            lay1_times_lay2_1=lay1_unkno_kno_spmat_list1*lay2_kno_unkno_spmat_list; % [(H1)'.*(E1)'] * E2
            lay1_times_lay2_2=lay1_unkno_kno_spmat_list*lay2_kno_unkno_spmat_list1; % (E1)' * [(E2).*(H2)]
%             lay1_times_lay2_3=lay1_unkno_kno_spmat_list1_2*lay2_kno_unkno_spmat_list_2; % [(L1)'.*(U1)'] * U2
%             lay1_times_lay2_4=lay1_unkno_kno_spmat_list_2*lay2_kno_unkno_spmat_list1_2; % (U1)' * [(U2).*(L2)]
%             lay1_times_lay2=lay1_times_lay2_1+lay1_times_lay2_2+(lay1_times_lay2_3+lay1_times_lay2_4).*eps;  %Matching matrix R*
            lay1_times_lay2=lay1_times_lay2_1+lay1_times_lay2_2; %Matching matrix R
            matched_unkno_num=0;
            lay_matched_unkno=zeros(unkno_num,2);

            themax=1;            
            while(themax~=0&&size(lay1_times_lay2,1)~=0&&size(lay1_times_lay2,2)~=0)
                %Calculate the max value
                themax=thre*max(max(lay1_times_lay2));  % Find the maximum value in R and get a set matching degree
                %Get every row and col's max value
                lay1_times_lay2(lay1_times_lay2<themax)=0; 
                [findmax_row(:,3),findmax_row(:,2)]=max(lay1_times_lay2,[],2);%Get every row's max: row num, col num, max value
                findmax_row(:,1)=1:size(findmax_row,1);
                [findmax_col(:,3),findmax_col(:,1)]=max(lay1_times_lay2,[],1);%Get every col's max: row num, col num, max value
                findmax_col(:,2)=1:size(findmax_col,1);
                %Get themax of all, remove others
                result_max_row=findmax_row.*(findmax_row>=themax);
                result_max_row(:,1:2)=0;
                findmax_row(all(result_max_row==0,2),:)=[];
                result_max_col=findmax_col.*(findmax_col>=themax);
                result_max_col(:,1:2)=0;
                findmax_col(all(result_max_col==0,2),:)=[];
                %Take intersection
                asso_resu_temp=intersect(findmax_row,findmax_col,'rows');  % Find the node pair with matching degree greater than the set value 
                if themax==0
                    clear findmax_row;clear findmax_col; clear result_max_row; clear result_max_col; %clear asso_resu_temp;         
                    break;
                end

                asso_num=0;
                asso_resu=zeros(size(asso_resu_temp,1),4);
                lay1_2_rela_kno_num_thisturn=size(lay1_2_rela_kno,1);  %Number of matched node pairs

                for i=1:size(asso_resu_temp,1)  %Traversing matched pairs of nodes
                        %findmax_row(i)
                        asso_num=asso_num+1;
                        matched_unkno_num=matched_unkno_num+1;
                        asso_resu(asso_num,1)=lay1_unkno_new_oldID(asso_resu_temp(i,1),2);   %Node original ID         
                        asso_resu(asso_num,2)=lay2_unkno_new_oldID(asso_resu_temp(i,2),2);   %Node original ID
                        asso_resu(asso_num,3)=asso_resu_temp(i,1); %Node new ID
                        asso_resu(asso_num,4)=asso_resu_temp(i,2); %Node new ID
                        lay_matched_unkno(matched_unkno_num,1)=asso_resu (asso_num,3); %Store the new IDs of all node pairs that will be matched in this round
                        lay_matched_unkno(matched_unkno_num,2)=asso_resu(asso_num,4);

                        lay1_2_rela_kno(asso_num+lay1_2_rela_kno_num_thisturn,1)=asso_resu(asso_num,1); %Store the original ID of all matching node pairs 
                        lay1_2_rela_kno(asso_num+lay1_2_rela_kno_num_thisturn,2)=asso_resu(asso_num,2);
                end
                clear findmax_row;clear findmax_col; clear result_max_row; clear result_max_col; %clear asso_resu_temp;           

                %Restart the calculation of the matching degree matrix R
                clear lay1_kno_new_oldID;clear lay2_kno_new_oldID; clear lay1_unkno_new_oldID; clear lay2_unkno_new_oldID; clear lay2_kno_new_oldID_sortnew;
                clear lay1_unkno_kno_spmat_list; clear lay2_kno_unkno_spmat_list; clear lay1_unknode_kedge_mat; clear lay2_kedge_unknode_mat;
                clear lay1_unkno_kno_spmat_list_2; clear lay2_kno_unkno_spmat_list_2;
                kno_num_thisturn=size(lay1_2_rela_kno,1);  %Number of matched node pairs
                lay1_2_rela_kno(:,3)=1:kno_num_thisturn;   
                lay1_kno_new_oldID=lay1_2_rela_kno(:,[3,1]);
                lay2_kno_new_oldID=lay1_2_rela_kno(:,[3,2]);
                
                lay1_unkno_new_oldID(:,1)=1:(size(lay1_linklist,1)-kno_num_thisturn);
                lay1_unkno_new_oldID(:,2)=setdiff(lay1_degree(:,1),lay1_kno_new_oldID(:,2)); 
                lay2_unkno_new_oldID(:,1)=1:(size(lay2_linklist,1)-kno_num_thisturn);
                lay2_unkno_new_oldID(:,2)=setdiff(lay2_degree(:,1),lay2_kno_new_oldID(:,2)); 
                lay1_unkno_new_oldID(:,3)=lay1_degree(lay1_unkno_new_oldID(:,2),2);
                lay2_unkno_new_oldID(:,3)=lay2_degree(lay2_unkno_new_oldID(:,2),2);

                lay2_kno_new_oldID_sortnew=sortrows(lay2_kno_new_oldID,1);
                %Generate two intermediate matrices E1, E2 (sub-matrix of adjacency matrix A) 
                lay1_unkno_kno_spmat_list=lay1_linklist(lay1_unkno_new_oldID(:,2),lay1_kno_new_oldID(:,2)); 
                lay2_kno_unkno_spmat_list=lay2_linklist(lay2_kno_new_oldID_sortnew(:,2),lay2_unkno_new_oldID(:,2));

%                 %Generate second-order intermediate matrix U1, U2 (sub-matrix of second-order adjacency matrix B) 
%                 lay1_unkno_kno_spmat_list_2=lay1_linklist_2(lay1_unkno_new_oldID(:,2),lay1_kno_new_oldID(:,2)); 
%                 lay2_kno_unkno_spmat_list_2=lay2_linklist_2(lay2_kno_new_oldID_sortnew(:,2),lay2_unkno_new_oldID(:,2));

                %Matrix dot product(H1)'.*(E1)'; (E2).*(H2)
                lay1_unkno_kno_spmat_list1=lay1_unkno_kno_spmat_list.*(repmat(1./(log(lay1_degree(lay1_kno_new_oldID(:,2),2)))',[size(lay1_unkno_new_oldID,1),1]));
                lay1_unkno_kno_spmat_list1(isnan(lay1_unkno_kno_spmat_list1))=0;
                lay1_unkno_kno_spmat_list1(isinf(lay1_unkno_kno_spmat_list1))=0;
                lay2_kno_unkno_spmat_list1=lay2_kno_unkno_spmat_list.*(repmat(1./(log(lay2_degree(lay2_kno_new_oldID_sortnew(:,2),2))),[1,size(lay2_unkno_new_oldID,1)]));
                lay2_kno_unkno_spmat_list1(isnan(lay2_kno_unkno_spmat_list1))=0;
                lay2_kno_unkno_spmat_list1(isinf(lay2_kno_unkno_spmat_list1))=0;

%                 %Matrix dot product (L1)'.*(U1)' ; (U2).*(L2)
%                 lay1_unkno_kno_spmat_list1_2=lay1_unkno_kno_spmat_list_2.*(repmat(1./(log(lay1_degree(lay1_kno_new_oldID(:,2),2)))',[size(lay1_unkno_new_oldID,1),1]));
%                 lay1_unkno_kno_spmat_list1_2(isnan(lay1_unkno_kno_spmat_list1_2))=0;
%                 lay1_unkno_kno_spmat_list1_2(isinf(lay1_unkno_kno_spmat_list1_2))=0;
%                 lay2_kno_unkno_spmat_list1_2=lay2_kno_unkno_spmat_list_2.*(repmat(1./(log(lay2_degree(lay2_kno_new_oldID_sortnew(:,2),2))),[1,size(lay2_unkno_new_oldID,1)]));
%                 lay2_kno_unkno_spmat_list1_2(isnan(lay2_kno_unkno_spmat_list1_2))=0;
%                 lay2_kno_unkno_spmat_list1_2(isinf(lay2_kno_unkno_spmat_list1_2))=0;

                %Matrix product
                lay1_times_lay2_1=lay1_unkno_kno_spmat_list1*lay2_kno_unkno_spmat_list; % [(H1)'.*(E1)'] * E2
                lay1_times_lay2_2=lay1_unkno_kno_spmat_list*lay2_kno_unkno_spmat_list1; % (E1)' * [(E2).*(H2)]
%                 lay1_times_lay2_3=lay1_unkno_kno_spmat_list1_2*lay2_kno_unkno_spmat_list_2; % [(L1)'.*(U1)'] * U2
%                 lay1_times_lay2_4=lay1_unkno_kno_spmat_list_2*lay2_kno_unkno_spmat_list1_2; % (U1)' * [(U2).*(L2)]
%                 lay1_times_lay2=lay1_times_lay2_1+lay1_times_lay2_2+(lay1_times_lay2_3+lay1_times_lay2_4).*eps;%Matching matrix R*
                lay1_times_lay2=lay1_times_lay2_1+lay1_times_lay2_2; %Matching matrix R
            end
            

            therownum=int8(known_rate/krIntval);
            %calculate precision and recall
            rightnum=0;
            size_kno_num=size(lay1_2_rela_kno,1);
            
            asso_resu=lay1_2_rela_kno(kno_num+1:size_kno_num,1:2);
            for i=1:size(asso_resu)
                if(ismember(asso_resu(i,:),lay1_2_rela_unkno,'rows')==1)
                    rightnum=rightnum+1;
                end
            end
            recall(therownum,experi_times)=rightnum/unkno_num;
            if size(asso_resu,1)==0
                 precision(therownum,experi_times)=0;
            else
                 precision(therownum,experi_times)=rightnum/size(asso_resu,1);
            end
                 edges_num(therownum,experi_times)=size(asso_resu,1);
            clear asso_resu;
            clear asso_resu_temp;
            clear lay1_2_rela_kno;
            clear lay1_2_rela_unkno;
            clear lay1_kno_new_oldID;
            clear lay1_times_lay2;
            clear lay1_unkno_kno_spmat_list;
            clear lay1_unkno_kno_spmat_list_2;
            clear lay1_unkno_new_oldID;
            clear lay2_kno_new_oldID;
            clear lay2_kno_new_oldID_sortnew;
            clear lay2_kno_unkno_spmat_list;
            clear lay2_kno_unkno_spmat_list_2;
            clear lay2_unkno_new_oldID;
            clear lay_matched_unkno;
        end
        
    end
    
end
output_args.recall=mean(recall,2);
output_args.precision=mean(precision,2);
output_args.f_measure=2*output_args.recall.*output_args.precision./(output_args.recall+output_args.precision);
output_args.edges_num=edges_num;