answer = menu('Qual o elemento que quer usar?','TRI3','TRI6','QUAD4','QUAD8');
% Handle response
switch answer
    case 1
        run('.\TRI3\TRI3.m')
    case 2
        run('.\TRI6\TRI6.m')
    case 3
        run('.\QUAD4\QUAD4.m')
    case 4
        run('.\QUAD8\QUAD8.m')
end