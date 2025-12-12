pkg load io;

% -------------------------------------------------
% Input
% -------------------------------------------------
x = xlsread('mesh_data_quad8.xlsx','coord','A1:A201') / 1000;
y = xlsread('mesh_data_quad8.xlsx','coord','B1:B201') / 1000;
quad8 = xlsread('mesh_data_quad8.xlsx','conec','A1:H52');

Nels = size(quad8,1);
Nnds = length(x);

% -------------------------------------------------
% Mesh plot
% -------------------------------------------------
figure(1); clf; hold on;
for i = 1:Nels
    edofs = quad8(i,[1 5 2 6 3 7 4 8]);
    plot(x(edofs), y(edofs), 'b');
end
plot(x,y,'ro');
axis equal;

% -------------------------------------------------
% Global system (SPARSE)
% -------------------------------------------------
Kg = sparse(Nnds, Nnds);
fg = zeros(Nnds,1);

% -------------------------------------------------
% Element assembly
% -------------------------------------------------
for i = 1:Nels
    edofs = quad8(i,:);
    XN = [x(edofs), y(edofs)];
    fL = 0;
    [Ke, fe] = Elem_Quad8(XN, fL);

    Kg(edofs,edofs) += Ke;
    fg(edofs) += fe;
end

% -------------------------------------------------
% Dirichlet BCs (elimination)
% -------------------------------------------------
exit_nodes = [33 30 34 201 199 200];
Kg(exit_nodes,:) = 0;
Kg(:,exit_nodes) = 0;
Kg(exit_nodes,exit_nodes) = speye(length(exit_nodes));
fg(exit_nodes) = 0;

% -------------------------------------------------
% Robin BCs
% -------------------------------------------------
p_robin = 0;
gama = 2.5;

robin_sides = {
    [19 7 6];
    [6 8 18];
    [18 115 116];
    [116 117 124]
};

for k = 1:length(robin_sides)
    ed = robin_sides{k};
    [He, Pe] = Robin_quadr( ...
        x(ed(1)), y(ed(1)), ...
        x(ed(2)), y(ed(2)), ...
        x(ed(3)), y(ed(3)), ...
        p_robin, gama);

    Kg(ed,ed) += He;
    fg(ed) += Pe;
end

% -------------------------------------------------
% Diagnostics
% -------------------------------------------------
disp(full(any(isnan(Kg(:)))));
disp(full(any(isinf(Kg(:)))));
disp(rank(Kg));
disp(condest(Kg));

% -------------------------------------------------
% Solve
% -------------------------------------------------
u = Kg \ fg;

% -------------------------------------------------
% Solution plot
% -------------------------------------------------
figure; clf; hold on;
for i = 1:Nels
    edofs = quad8(i,[1 5 2 6 3 7 4 8]);
    fill3(x(edofs), y(edofs), u(edofs), u(edofs));
end
view(3); colorbar; axis equal;

% -------------------------------------------------
% Post-processing
% -------------------------------------------------
abs_vel_nds = zeros(Nnds,1);
abs_vel = zeros(Nels,1);        % <<< FIX
pressure = zeros(Nels,1);       % <<< FIX
vel = zeros(Nels,2);

figure; hold on;
for i = 1:Nels
    edofs = quad8(i,:);
    XN = [x(edofs), y(edofs)];

    nip = 4;
    [xp, ~] = Genip2DQ(nip);
    abs_vel_ip = zeros(nip,1);

    for ip = 1:nip
        [B, psi] = Shape_N_Der8(XN, xp(ip,1), xp(ip,2));
        gradu = B' * u(edofs);

        vel(i,1) = gradu(1);
        vel(i,2) = gradu(2);
        abs_vel_ip(ip) = norm(gradu);
    end

    abs_vel(i) = mean(abs_vel_ip);
    pressure(i) = 101328.8281 - 0.6125 * abs_vel(i)^2;
    abs_vel_nds(edofs) = abs_vel(i);
end

% -------------------------------------------------
% Global extrema
% -------------------------------------------------
p_field = 101328.8281 - 0.6125 * abs_vel_nds.^2;

Pmax  = max(p_field);
Pmin  = min(p_field);
Umax  = max(abs_vel_nds);
Umin  = min(abs_vel_nds);
POTmax = max(u);

% -------------------------------------------------
% Excel export (exact MATLAB-style format)
% -------------------------------------------------
outfile = 'Results_quad8_V2.xlsx';

headers = {
    '#NODE', 'VELOCITY POTENTIAL', '', ...
    '#ELEMENT', 'U m/s (x  vel)', 'V m/s (y vel)', '|V| m/s', 'Pressure (Pa)'
};
xlswrite(outfile, headers, 1, 'A1');

% Nodal block (A–B)
node_id = (1:Nnds)';
nodal_block = [node_id, u(:)];
xlswrite(outfile, nodal_block, 1, 'A2');

% Element block (D–H)
elem_id = (1:Nels)';
element_block = [ ...
    elem_id, ...
    vel(:,1), ...
    vel(:,2), ...
    abs_vel(:), ...
    pressure(:) ...
];

xlswrite(outfile, element_block, 1, 'D2');
