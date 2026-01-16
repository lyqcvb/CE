clear
clc
close
N=16;
dirout=dir(fullfile('C:\Users\Lenovo\Desktop\课题组\最终数据\AD\AD -256 100s 分节律\beta'));
filename={dirout.name};
for i=3:31
    F=cell2mat(filename(1,i));
    name=F(1:end-4);
    file = strcat('C:\Users\Lenovo\Desktop\课题组\最终数据\AD\AD -256 100s 分节律\beta\');
    shuju=strcat(F);
    x=[file,shuju];
    load(x);
    for ii = 1:16
        % 提取第i个通道的信号
        signal = eegdata(ii, 1:25601);
        
        % 应用希尔伯特变换
        analytic_signal = hilbert(signal);
        % 计算包络
        envelope(ii,:) = abs(analytic_signal);
    end
    x=envelope;
    y=envelope;
    A=[]; 
    B=[];
    C=[];
    for  p=1:16
        X=x(p,:);
        for  q=1:16    
            Y=y(q,:);
            r=corr2(X,Y);
        A(p,q)=r;
        end    
    end
    A3=abs(A);

    [B3] = Remove_selfcon(A3,A3);%高阶

    [C3]=Remove_selfcon_again(A3,B3);




    A33(:,:,i-2)=A3;
    B33(:,:,i-2)=B3;
    C33(:,:,i-2)=C3;

    A3=erzhihua(A3,0.35);
    B3=erzhihua(B3,0.35);
    C3=erzhihua(C3,0.35);

    Eg1=efficiency_bin(A3);%全局效率
    Eg2=efficiency_bin(B3);%全局效率
    Eg3=efficiency_bin(C3);%全局效率

    DC=sum(A3, 2);
    DC1=sum(B3, 2);
    DC2=sum(C3, 2) + sum(C3, 1)';
    DC=DC.';
    DC1=DC1.';
    DC2=DC2.';

    BC=betweenness_bin(A3);
    BC1=betweenness_bin(B3); 
    BC2=betweenness_bin(C3);

    [D] = distance_bin(A3);
    CCc = 1 ./ sum(D, 2); % 计算每个节点的接近中心性
    [D1] = distance_bin(B3);
    CC1 = 1 ./ sum(D1, 2); % 计算每个节点的接近中心性
    [D2] = distance_bin(C3);
    CC2 = 1 ./ sum(D2, 2); % 计算每个节点的接近中心性
    CCc=CCc.';
    CC1=CC1.';
    CC2=CC2.';

    bbc=sqrt(DC.^2+BC.^2+CCc.^2);
    bbc1=sqrt(DC1.^2+BC1.^2+CC1.^2);
    bbc2=sqrt(DC2.^2+BC2.^2+CC2.^2);
    for hub=1:16
        bbcc=bbc;bbcc1=bbc1;bbcc2=bbc2;
        bbcc(hub)=[];
        bbcc1(hub)=[];
        bbcc2(hub)=[];
        bbcc = bbcc/ sum(bbcc);
        % 计算熵
        bc(:,hub) = -sum(bbc(hub) .* log(bbcc));

        bbcc1  = bbcc1/ sum(bbcc1);
        % 计算熵
        bc1(:,hub)= -sum(bbc1(hub).* log(bbcc1));

        bbcc2  = bbcc2/ sum(bbcc2);
        % 计算熵
        bc2(:,hub) = -sum(bbc2(hub) .* log(bbcc2));

    end
    [W,Q]=sort(bbc,2,"descend");
    [W1,Q1]=sort(bbc1,2,"descend");  
    [W2,Q2]=sort(bbc2,2,"descend");

    meanVal = mean(bc);
    stdVal = std(bc);
    threshold = meanVal + stdVal;
    degree(i-2,:) = (bc > threshold);


    meanVal1 = mean(bc1);
    stdVal1 = std(bc1);
    threshold1 = meanVal1 + stdVal1;
    degree1(i-2,:) = (bc1 > threshold1);

    meanVal2 = mean(bc2);
    stdVal2 = std(bc2);
    threshold2 = meanVal2 + stdVal2;
    degree2(i-2,:) = (bc2 > threshold2);





    AAA=A3;BBB=B3;CCC=C3;
    for w=1:16
    QQ=Q(1,w);
        AAA(:,QQ)=0;
        AAA(QQ,:)=0;
        A333(:,:,w)=AAA;
    QQ1=Q1(1,w);
        BBB(:,QQ1)=0;
        BBB(QQ1,:)=0;
        B333(:,:,w)=BBB;
    QQ2=Q2(1,w);
        CCC(:,QQ)=0;
        CCC(QQ,:)=0;
        C333(:,:,w)=CCC;
    end
    for ww=1:16
        AA=A333(:,:,ww);
        Eglob(:,ww)=efficiency_bin(AA)/Eg1;
        BB=B333(:,:,ww);
        Eglob1(:,ww)=efficiency_bin(BB)/Eg2;
        CC=C333(:,:,ww);
        Eglob2(:,ww)=efficiency_bin(CC)/Eg3;
    end


    DTF1(i-2,:)=mean(A3,"all");
    DTF2(i-2,:)=mean(B3,"all");
    DTF3(i-2,:)=mean(C3,"all");

    EE(i-2,:)=Eglob;
    EE1(i-2,:)=Eglob1;
    EE2(i-2,:)=Eglob2;
end
A333=mean(A33,3);
B333=mean(B33,3);
C333=mean(C33,3);

EEE=mean(EE,1);
EEE1=mean(EE1,1);
EEE2=mean(EE2,1);

degree=mean(degree);
degree1=mean(degree1);
degree2=mean(degree2);
[WW,QW]=sort(degree,2,"descend");
[WW1,QW1]=sort(degree1,2,"descend");  
[WW2,QW2]=sort(degree2,2,"descend");

%%
Eglob1(i-2,:)=efficiency_wei(A3);%全局效率
Eglob2(i-2,:)=efficiency_wei(B3);%全局效率
Eglob3(i-2,:)=efficiency_wei(C3);%全局效率

AA1(i-2,:)=mean(clustering_coef_wu(A3));
AA2(i-2,:)=mean(clustering_coef_wu(B3));
AA3(i-2,:)=mean(clustering_coef_wd(C3));

bc1(i-2,:)=2*betweenness_wei(A3)/((N-1)*(N-2));%介数
bc2(i-2,:)=2*betweenness_wei(A3)/((N-1)*(N-2));%介数
bc3(i-2,:)=2*betweenness_wei(A3)/((N-1)*(N-2));%介数

[Ci1, q1(i-2,:)]=modularity_louvain_dir(A3);%模块度
[Ci2, q2(i-2,:)]=modularity_louvain_dir(B3);%模块度
[Ci3, q3(i-2,:)]=modularity_louvain_dir(C3);%模块度
PC1(i-2,:)=pc(A3,Ci1);PC1=PC1.';%参与系数
PC2=pc(B3,Ci2);PC2=PC2.';%参与系数
PC3=pc(C3,Ci3);PC3=PC3.';%参与系数


%%
I1=erzhihua(A3,0.4);
I2=erzhihua(B3,0.4);
I3=erzhihua(C3,0.4);
[RLOFC] = rich_club_bu(I1,16);%富人俱乐部
[RHOFC] = rich_club_bu(I2,16);%富人俱乐部
[RaHOFC] = rich_club_bd(I3,32);%富人俱乐部
for Ji=1:1000
[R1 ,eff1] = randmio_und_connected(I1, 2);
[RAN1] = rich_club_bu(R1,16);%富人俱乐部
wer1(:,:,Ji)=RAN1;
[R2 ,eff2] = randmio_und_connected(I2, 2);
[RAN2] = rich_club_bu(R2,16);%富人俱乐部 
wer2(:,:,Ji)=RAN2;
[R3 ,eff3] = randmio_dir_connected(I3, 2);
[RAN3] = rich_club_bd(R3,32);%富人俱乐部
wer3(:,:,Ji)=RAN3;
end
ranA=mean(wer1,3);
ranB=mean(wer2,3);
ranC=mean(wer3,3);
rc1(i-2,:)=RLOFC./ranA;

rc2(i-2,:)=RHOFC./ranB;

rc3(i-2,:)=RaHOFC./ranC;

%%
bc=betweenness_wei(A3);
bc1=betweenness_wei(B3); 
bc2=betweenness_wei(C3);
bc=bc.';
bc1=bc1.';
bc2=bc2.';

meanVal = mean(bc);
stdVal = std(bc);
threshold = meanVal + stdVal;
BC(i-2,:) = (bc > threshold);

meanVal1 = mean(bc1);
stdVal1 = std(bc1);
threshold1 = meanVal1 + stdVal1;
BC1(i-2,:) = (bc1 > threshold1);

meanVal2 = mean(bc2);
stdVal2 = std(bc2);
threshold2 = meanVal2 + stdVal2;
BC2(i-2,:) = (bc2 > threshold2);


%%



