function Ao = softmax( Ai )
max_val = max(Ai(:));
Ai = Ai - max_val;

sm_exp = exp(Ai);
Ao = sm_exp ./ sum(sm_exp);
end

