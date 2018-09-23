function [result]=check_pca()
    test_nu=77;
    nShape = 47;
    nVerts = 11510;
    nFaces = 11540;
    test_num = 77;
    iden_num = 50;
    X=zeros(test_num, nShape*nVerts * 3);
    filename = 'D:/sydney/first/data/tester_ (';
	for i = 1:test_num
		name = [filename , mat2str(i) , ')/Blendshape/shape.bs']
		fp=fopen(name, 'rb');
		nShapes = 0;
        nVerts = 0;
        nFaces = 0;
		[nShapes]=fread(fp,1,'int32')	      % nShape = 46
		[nVerts]=fread(fp,1,'int32')		%	// nVerts = 11510
		[nFaces]=fread(fp,1,'int32')		%	// nFaces = 11540\
        
		%Load neutral expression B_0
		for j = 1 : nVerts * 3
            X(i,j)=fread(fp,1,'single');
        end

		% Load other expressions B_i ( 1 <= i <= 46 )
		for exprId = 1 : nShapes
			for j = 1 : nVerts * 3
				X(i, 3 * nVerts*exprId + j) = fread(fp,1,'single');
            end
        end
        
		
		fclose(fp);
    end
    %V=pca(X');
    x_m=mean(X,2);
    x_m

%     size(X)
%     size(x_m)
    X_wc=X-repmat(x_m,[1,3*nVerts*nShape]);
    A=X_wc*X_wc';
    A=A./(3*nVerts*nShape);
    [S V D]=svd(A);
    V
    V(1,1)
    V(2,2)
    V(3,3)
    result = D(1:iden_num,:)'*X;



%     result=(X'*V(:,1:iden_num))';
%     result(1:10,1:10)
    %save(result,'svd_50.mat');
end