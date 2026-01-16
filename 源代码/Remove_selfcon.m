function  [HOFC] = Remove_selfcon(in1, in2)

n1=length(in1);

for i=1:1:n1-1
    for j=i+1:1:n1  %j=i，因为不同频段的耦合需要同一通道编号进行计算
        a=in1(:,i);
        b=in2(:,j);  
        a(i,:)=[];
        b(j,:)=[];
        
        j1=j-1;                                                                                                                                                                                                                                                                              
        a(j1,:)=[];
        b(i,:)=[];

        a1=0.5*[log(1+a)-log(1-a)];
        b1=0.5*[log(1+b)-log(1-b)];
        HOFC(i,j)=corr(b1,a1);
        HOFC(j,i)=HOFC(i,j);
        HOFC=abs(HOFC);
    end
end
