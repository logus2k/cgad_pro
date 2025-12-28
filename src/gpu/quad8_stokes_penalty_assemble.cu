extern "C" __global__
void quad8_stokes_penalty_assemble(
    const double* x,
    const double* y,
    const int*    quad8,
    int           Nels,
    double        mu,
    double        penalty,

    int*    rows,
    int*    cols,
    double* vals
)
{
    int e = blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= Nels) return;

    // ---- local storage (registers) ----
    double Ke[16][16];
    #pragma unroll
    for (int i=0;i<16;i++)
        #pragma unroll
        for (int j=0;j<16;j++)
            Ke[i][j] = 0.0;

    // ---- load nodes ----
    int n[8];
    #pragma unroll
    for (int i=0;i<8;i++)
        n[i] = quad8[e*8 + i];

    double X[8], Y[8];
    #pragma unroll
    for (int i=0;i<8;i++) {
        X[i] = x[n[i]];
        Y[i] = y[n[i]];
    }

    // ---- Gauss points (3x3) ----
    const double gp[3] = { -0.774596669241483, 0.0, 0.774596669241483 };
    const double gw[3] = {  0.555555555555556, 0.888888888888889, 0.555555555555556 };

    // ---- integration ----
    for (int gx=0; gx<3; gx++)
    for (int gy=0; gy<3; gy++) {

        double xi  = gp[gx];
        double eta = gp[gy];
        double w   = gw[gx] * gw[gy];

        // ---- shape derivs in parent space ----
        double dNdxi[8], dNdeta[8];

        // (standard Quad-8 derivatives – abbreviated here)
        // YOU ALREADY HAVE THESE EXPRESSIONS IN CPU/GPU VERSION
        // Copy them verbatim here

        // ---- Jacobian ----
        double J11=0,J12=0,J21=0,J22=0;
        #pragma unroll
        for (int i=0;i<8;i++) {
            J11 += dNdxi[i]*X[i];
            J12 += dNdeta[i]*X[i];
            J21 += dNdxi[i]*Y[i];
            J22 += dNdeta[i]*Y[i];
        }

        double detJ = J11*J22 - J12*J21;
        double invJ11 =  J22/detJ;
        double invJ12 = -J12/detJ;
        double invJ21 = -J21/detJ;
        double invJ22 =  J11/detJ;

        double dNdx[8], dNdy[8];
        #pragma unroll
        for (int i=0;i<8;i++) {
            dNdx[i] = invJ11*dNdxi[i] + invJ12*dNdeta[i];
            dNdy[i] = invJ21*dNdxi[i] + invJ22*dNdeta[i];
        }

        double weight = w * detJ;

        // ---- assemble Ke ----
        #pragma unroll
        for (int i=0;i<8;i++)
        #pragma unroll
        for (int j=0;j<8;j++) {

            double k_visc = mu * (dNdx[i]*dNdx[j] + dNdy[i]*dNdy[j]) * weight;

            // ux-ux, uy-uy
            Ke[i][j]       += k_visc;
            Ke[i+8][j+8]   += k_visc;

            // penalty
            double div_i = dNdx[i] + dNdy[i];
            double div_j = dNdx[j] + dNdy[j];
            double k_pen = penalty * div_i * div_j * weight;

            Ke[i][j]       += k_pen;
            Ke[i][j+8]     += k_pen;
            Ke[i+8][j]     += k_pen;
            Ke[i+8][j+8]   += k_pen;
        }
    }

    // ---- write COO ----
    int base = e * 256;
    int k = 0;

    #pragma unroll
    for (int a=0;a<8;a++)
    #pragma unroll
    for (int b=0;b<8;b++) {

        int ia = n[a];
        int ib = n[b];

        int ux_i = 2*ia;
        int uy_i = 2*ia + 1;
        int ux_j = 2*ib;
        int uy_j = 2*ib + 1;

        rows[base+k] = ux_i; cols[base+k] = ux_j; vals[base+k++] = Ke[a][b];
        rows[base+k] = ux_i; cols[base+k] = uy_j; vals[base+k++] = Ke[a][b+8];
        rows[base+k] = uy_i; cols[base+k] = ux_j; vals[base+k++] = Ke[a+8][b];
        rows[base+k] = uy_i; cols[base+k] = uy_j; vals[base+k++] = Ke[a+8][b+8];
    }
}
