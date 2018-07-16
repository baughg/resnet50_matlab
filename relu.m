function Ao = relu( Ai )
Ao = Ai;
Ao(Ai < 0) = 0;
end

