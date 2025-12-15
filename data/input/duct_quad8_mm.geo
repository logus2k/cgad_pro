SetFactory("OpenCASCADE");

// =================================================
// PARAMETERS (mm)
// =================================================
L  = 3000;
H  = 1200;

xm = 1250;        // center of top/bottom reliefs
Rr = 250;         // radius of top/bottom reliefs

Lbranch = 1000;   // branch length
Hbranch = 600;    // branch height
Rbranch = Hbranch/2;

lc = 3;

// =================================================
// BASE DOMAIN
// =================================================
Rectangle(1) = {0, 0, 0, L, H};

// =================================================
// TOP / BOTTOM RELIEFS (full disks for Boolean)
// =================================================
Disk(10) = {xm, H, 0, Rr, Rr};
Disk(11) = {xm, 0, 0, Rr, Rr};

// =================================================
// RIGHT BIFURCATION (rectangle + rounded end)
// =================================================

// Rectangular part of branch
Rectangle(20) = {L - Lbranch, H/2 - Hbranch/2, 0,
                 Lbranch, Hbranch};

// Rounded end of branch
Disk(21) = {L - Lbranch, H/2, 0, Rbranch, Rbranch};

// =================================================
// BOOLEAN CUT
// =================================================
BooleanDifference(100) =
{
  Surface{1}; Delete;
}
{
  Surface{10, 11, 20, 21}; Delete;
};

// =================================================
// MESH SETTINGS (TRUE QUAD8)
// =================================================
Mesh.ElementOrder = 2;
Mesh.SecondOrderIncomplete = 0;
Mesh.RecombineAll = 1;

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Recombine Surface{:};
