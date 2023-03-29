function [Y,p1] = ms_scenario(Y,varargin)
% Missing data scenario - random missing & fiber missing.

s = size(Y);
ip = inputParser;
ip.addParameter('ms','fiber',@(x)ismember(x,{'random','fiber','bm'}));
ip.addParameter('missing_rate',0.2,@isscalar);
ip.parse(varargin{:});

ms = ip.Results.ms;
ratio = ip.Results.missing_rate;

if strcmp(ms,'random')
    load random_tensor;
    W = round(random_tensor+0.5-ratio);
  
elseif strcmp(ms,'fiber')
    load random_matrix;
    W = zeros(s);
    for i1 = 1:s(1)
        for i2 = 1:s(2)
            W(i1,i2,:) = round(random_matrix(i1,i2)+0.5-ratio);
        end
    end
    
else strcmp(ms,'bm')
    load BMrandom_tensor;
    W = round(BMrandom_tensor+0.5-ratio);
end
count=length(find(W==0));
 n = floor(count);
    f = find(Y);
  W1=reshape(double(W),[],1);
  
    p1 = find(W1==0);
    p1 = sort (p1, 'descend');
    %p2 = p1';
   
   
        
        for i=1:length(p1)
            Y(f(p1(i),1),f(p1(i),2),f(p1(i),3)) =0;
            
       
        end

end