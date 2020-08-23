function [accuracy]=my_face_recognition( train_dir,test_dir,train_num,test_num,energy,~ )
%该函数实现了利用PCA方法进行人脸识别的过程
%Input
%       train_dir：库数据集的目录
%       test_dir：测试数据集的目录
%       train_num：选择的库数据集的个数
%       test_num：要测试的数据集的个数，要小于库数据集个数

if train_num<=test_num
    fprintf('库数据集要大于测试数据集！\n');
    return ;
end


%因为文件大小固定，所以在此我们设置矩阵的行列为定值
row=315;
column=236;
train_data=zeros(train_num,row*column);%预分配数据可以加速数据读取，矩阵的行数是库数据的个数，列数是图片的维度
train_files=dir(train_dir);%获取库目录下的所有文件，获得的每一个文件都是一个结构体，我们需要的是其中的name属性。第一个和第二个文件分别表示当前目录和父目录，需要跳过
for i=1:train_num
    file_name=sprintf('%s\\%s',train_dir,train_files(i+2).name);%这里需要加双斜杠
    img_data = imread(file_name);
    [m,n] = size(img_data);
    if m ~= row || n ~= column
        img_data = imresize(img_data,[row column]);
    end
    %[row column]=size(img_data);
    img_data=img_data(1:row*column);%将读取的数据转成一个行向量
    train_data(i,:)=img_data;%将该行向量添加到库集中
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%求平均脸，与人脸识别无关，只是一个测试
imgmean=mean(train_data);
size(imgmean);
mean_img=reshape(imgmean,row,column);
mean_img=uint8(mean_img);
imwrite(mean_img,'H:\1.bmp');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
if b==1
    for i=1:test_num
        train_data(i,:)=train_data(i,:)-imgmean;
    end
end
%}

%进行主成份分析，返回的结果为
%   COEFF：特征向量
%   latent：特征值，按由大到小的顺序排列
%当数据的维度大于数据个数时，通过在函数后面添加参数‘econ’可以加速计算
%[COEFF,~,latent] = princomp(train_data,'econ');

[COEFF,~,latent] = pca(train_data);

%保存的维度（特征值）个数使图像保存的能量大于95%
dimension_left=0;
cum_percent=cumsum(latent)/sum(latent);
for i=1:length(cum_percent)
    if cum_percent(i)>=energy
        dimension_left=i;
        break;
    end
end
%fprintf('dimension left is %d\n',dimension_left);


%将库数据集进行降维
train_data_reduced=train_data*COEFF(:,1:dimension_left);

%读取测试数据集
test_data=zeros(train_num,row*column);%预分配数据可以加速数据读取
test_files=dir(test_dir);%获取库目录下的所有文件，获得的每一个文件都是一个结构体，我们需要的是其中的name属性。第一个和第二个文件分别表示当前目录和父目录，需要跳过
for i=1:test_num
    file_name=sprintf('%s\\%s',test_dir,test_files(i+2).name);%这里需要加双斜杠
    
    img_data = imread(file_name);
    [m,n] = size(img_data);
    if m ~= row || n ~= column
        img_data = imresize(img_data,[row column]);
    end
    img_data=img_data(1:row*column);%将读取的数据转成一个行向量
    test_data(i,:)=img_data;%将该行向量添加到库集中
end

%{
if b==1
    for i=1:test_num
        test_data(i,:)=test_data(i,:)-imgmean;
    end
end
%}

%将测试数据集进行降维
test_data_reduced=test_data*COEFF(:,1:dimension_left);

accuracy=0;
for i=1:test_num
    %通过计算向量二阶范数的方法计算欧式距离
    min=norm(test_data_reduced(i,:)-train_data_reduced(1,:));
    position=1;
    for j=2:train_num
        distance=norm(test_data_reduced(i,:)-train_data_reduced(j,:));
        if min>distance
            min=distance;
            position=j;
        end
    end
    %fprintf('test_file:%s,train_file;%s\n',test_files(i+2).name,train_files(position+2).name);
    if same_person(test_files(i+2).name,train_files(position+2).name)==1
        accuracy=accuracy+1;
    else
        %fprintf('test_file:%s,train_file;%s\n',test_files(i+2).name,train_files(position+2).name);
    end
end
accuracy=accuracy/test_num;
fprintf('Accuracy is %f,energy %f,dimension left %d\n',accuracy,energy,dimension_left);

%用来比较两个字符串的前五位是否相同
    function result=same_person(s1,s2)
        result=strncmp(s1,s2,100);
    end
end

