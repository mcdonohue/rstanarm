real mean_PPD[M];
real yAlpha1[intercept_type[1] > 0];
real yAlpha2[intercept_type[2] > 0];
real yAlpha3[intercept_type[3] > 0];
real yAlpha4[intercept_type[4] > 0];
real yAlpha5[intercept_type[5] > 0];
real yAlpha6[intercept_type[6] > 0];
real yAlpha7[intercept_type[7] > 0];
real yAlpha8[intercept_type[8] > 0];
real yAlpha9[intercept_type[9] > 0];
real yAlpha10[intercept_type[10] > 0];
real yAlpha11[intercept_type[11] > 0];
real yAlpha12[intercept_type[12] > 0];
real yAlpha13[intercept_type[13] > 0];
real yAlpha14[intercept_type[14] > 0];
real yAlpha15[intercept_type[15] > 0];
real yAlpha16[intercept_type[16] > 0];
real yAlpha17[intercept_type[17] > 0];
real yAlpha18[intercept_type[18] > 0];
real yAlpha19[intercept_type[19] > 0];
real yAlpha20[intercept_type[20] > 0];
vector[prior_dist_for_cov == 2 && bK1 > 0 ? size(bCov1_idx) : 0] bCov1;
vector[prior_dist_for_cov == 2 && bK2 > 0 ? size(bCov2_idx) : 0] bCov2;
vector[bN1 * bK1] b1 = to_vector(bMat1'); // ensures same order as stan_glmer (make_b)
vector[bN2 * bK2] b2 = to_vector(bMat2');

matrix[yNeta[1],yK[1]] yXlt1; // fe design matrix with latent time shift
matrix[yNeta[2],yK[2]] yXlt2;
matrix[yNeta[3],yK[3]] yXlt3;
matrix[yNeta[4],yK[4]] yXlt4;
matrix[yNeta[5],yK[5]] yXlt5;
matrix[yNeta[6],yK[6]] yXlt6;
matrix[yNeta[7],yK[7]] yXlt7;
matrix[yNeta[8],yK[8]] yXlt8;
matrix[yNeta[9],yK[9]] yXlt9;
matrix[yNeta[10],yK[10]] yXlt10;
matrix[yNeta[11],yK[11]] yXlt11;
matrix[yNeta[12],yK[12]] yXlt12;
matrix[yNeta[13],yK[13]] yXlt13;
matrix[yNeta[14],yK[14]] yXlt14;
matrix[yNeta[15],yK[15]] yXlt15;
matrix[yNeta[16],yK[16]] yXlt16;
matrix[yNeta[17],yK[17]] yXlt17;
matrix[yNeta[18],yK[18]] yXlt18;
matrix[yNeta[19],yK[19]] yXlt19;
matrix[yNeta[20],yK[20]] yXlt20;

vector[yK[1]] yXltbar1; // predictor (with latent time) means
vector[yK[2]] yXltbar2;
vector[yK[3]] yXltbar3;
vector[yK[4]] yXltbar4;
vector[yK[5]] yXltbar5;
vector[yK[6]] yXltbar6;
vector[yK[7]] yXltbar7;
vector[yK[8]] yXltbar8;
vector[yK[9]] yXltbar9;
vector[yK[10]] yXltbar10;
vector[yK[11]] yXltbar11;
vector[yK[12]] yXltbar12;
vector[yK[13]] yXltbar13;
vector[yK[14]] yXltbar14;
vector[yK[15]] yXltbar15;
vector[yK[16]] yXltbar16;
vector[yK[17]] yXltbar17;
vector[yK[18]] yXltbar18;
vector[yK[19]] yXltbar19;
vector[yK[20]] yXltbar20;

// Evaluate mean_PPD
{
  int bMat1_colshift = 0; // column shift in bMat1
  int bMat2_colshift = 0; // column shift in bMat2

  // Linear predictor for submodel 1
  if (M > 0) {
    vector[yNeta[1]] yEta1;
    for(i in 1:yK[1]){
      yXlt1[,i] = yX1[,i];
    }
    for(j in 1:yNeta[1]){
      yXlt1[j,lt_idx[1]] = yX1[j,lt_idx[1]] + lt[y1_Z1_id[j]];
    }
    yEta1 = evaluate_mu(evaluate_eta(yXlt1, y1_Z1, y1_Z2, y1_Z1_id, y1_Z2_id, yGamma1, yBeta1,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[1], y1_offset), 
                        family[1], link[1]);
    mean_PPD[1] = mean_PPD_rng(yEta1, yAux1, family[1]);
  }

  // Linear predictor for submodel 2
  if (M > 1) {
    vector[yNeta[2]] yEta2;
    bMat1_colshift += bK1_len[1];
    bMat2_colshift += bK2_len[1];
    for(i in 1:yK[2]){
      yXlt2[,i] = yX2[,i];
    }
    for(j in 1:yNeta[2]){
      yXlt2[j,lt_idx[2]] = yX2[j,lt_idx[2]] + lt[y2_Z1_id[j]];
    }
    yEta2 = evaluate_mu(evaluate_eta(yXlt2, y2_Z1, y2_Z2, y2_Z1_id, y2_Z2_id, yGamma2, yBeta2,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[2], y2_offset), 
                        family[2], link[2]);
    mean_PPD[2] = mean_PPD_rng(yEta2, yAux2, family[2]);
  }

  // Linear predictor for submodel 3
  if (M > 2) {
    vector[yNeta[3]] yEta3;
    bMat1_colshift += bK1_len[2];
    bMat2_colshift += bK2_len[2];
    for(i in 1:yK[3]){
      yXlt3[,i] = yX3[,i];
    }
    for(j in 1:yNeta[3]){
      yXlt3[j,lt_idx[3]] = yX3[j,lt_idx[3]] + lt[y3_Z1_id[j]];
    }
    yEta3 = evaluate_mu(evaluate_eta(yXlt3, y3_Z1, y3_Z2, y3_Z1_id, y3_Z2_id, yGamma3, yBeta3,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[3], y3_offset), 
                        family[3], link[3]);
    mean_PPD[3] = mean_PPD_rng(yEta3, yAux3, family[3]);
  }

  // Linear predictor for submodel 4
  if (M > 3) {
    vector[yNeta[4]] yEta4;
    bMat1_colshift += bK1_len[3];
    bMat2_colshift += bK2_len[3];
    for(i in 1:yK[4]){
      yXlt4[,i] = yX4[,i];
    }
    for(j in 1:yNeta[4]){
      yXlt4[j,lt_idx[4]] = yX4[j,lt_idx[4]] + lt[y4_Z1_id[j]];
    }
    yEta4 = evaluate_mu(evaluate_eta(yXlt4, y4_Z1, y4_Z2, y4_Z1_id, y4_Z2_id, yGamma4, yBeta4,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[4], y4_offset), 
                        family[4], link[4]);
    mean_PPD[4] = mean_PPD_rng(yEta4, yAux4, family[4]);
  }

  // Linear predictor for submodel 5
  if (M > 4) {
    vector[yNeta[5]] yEta5;
    bMat1_colshift += bK1_len[4];
    bMat2_colshift += bK2_len[4];
    for(i in 1:yK[5]){
      yXlt5[,i] = yX5[,i];
    }
    for(j in 1:yNeta[5]){
      yXlt5[j,lt_idx[5]] = yX5[j,lt_idx[5]] + lt[y5_Z1_id[j]];
    }
    yEta5 = evaluate_mu(evaluate_eta(yXlt5, y5_Z1, y5_Z2, y5_Z1_id, y5_Z2_id, yGamma5, yBeta5,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[5], y5_offset), 
                        family[5], link[5]);
    mean_PPD[5] = mean_PPD_rng(yEta5, yAux5, family[5]);
  }

  // Linear predictor for submodel 6
  if (M > 5) {
    vector[yNeta[6]] yEta6;
    bMat1_colshift += bK1_len[5];
    bMat2_colshift += bK2_len[5];
    for(i in 1:yK[6]){
      yXlt6[,i] = yX6[,i];
    }
    for(j in 1:yNeta[6]){
      yXlt6[j,lt_idx[6]] = yX6[j,lt_idx[6]] + lt[y6_Z1_id[j]];
    }
    yEta6 = evaluate_mu(evaluate_eta(yXlt6, y6_Z1, y6_Z2, y6_Z1_id, y6_Z2_id, yGamma6, yBeta6,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[6], y6_offset), 
                        family[6], link[6]);
    mean_PPD[6] = mean_PPD_rng(yEta6, yAux6, family[6]);
  }

  // Linear predictor for submodel 7
  if (M > 6) {
    vector[yNeta[7]] yEta7;
    bMat1_colshift += bK1_len[6];
    bMat2_colshift += bK2_len[6];
    for(i in 1:yK[7]){
      yXlt7[,i] = yX7[,i];
    }
    for(j in 1:yNeta[7]){
      yXlt7[j,lt_idx[7]] = yX7[j,lt_idx[7]] + lt[y7_Z1_id[j]];
    }
    yEta7 = evaluate_mu(evaluate_eta(yXlt7, y7_Z1, y7_Z2, y7_Z1_id, y7_Z2_id, yGamma7, yBeta7,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[7], y7_offset), 
                        family[7], link[7]);
    mean_PPD[7] = mean_PPD_rng(yEta7, yAux7, family[7]);
  }

  // Linear predictor for submodel 8
  if (M > 7) {
    vector[yNeta[8]] yEta8;
    bMat1_colshift += bK1_len[7];
    bMat2_colshift += bK2_len[7];
    for(i in 1:yK[8]){
      yXlt8[,i] = yX8[,i];
    }
    for(j in 1:yNeta[8]){
      yXlt8[j,lt_idx[8]] = yX8[j,lt_idx[8]] + lt[y8_Z1_id[j]];
    }
    yEta8 = evaluate_mu(evaluate_eta(yXlt8, y8_Z1, y8_Z2, y8_Z1_id, y8_Z2_id, yGamma8, yBeta8,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[8], y8_offset), 
                        family[8], link[8]);
    mean_PPD[8] = mean_PPD_rng(yEta8, yAux8, family[8]);
  }

  // Linear predictor for submodel 9
  if (M > 8) {
    vector[yNeta[9]] yEta9;
    bMat1_colshift += bK1_len[8];
    bMat2_colshift += bK2_len[8];
    for(i in 1:yK[9]){
      yXlt9[,i] = yX9[,i];
    }
    for(j in 1:yNeta[9]){
      yXlt9[j,lt_idx[9]] = yX9[j,lt_idx[9]] + lt[y9_Z1_id[j]];
    }
    yEta9 = evaluate_mu(evaluate_eta(yXlt9, y9_Z1, y9_Z2, y9_Z1_id, y9_Z2_id, yGamma9, yBeta9,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[9], y9_offset), 
                        family[9], link[9]);
    mean_PPD[9] = mean_PPD_rng(yEta9, yAux9, family[9]);
  }

  // Linear predictor for submodel 10
  if (M > 9) {
    vector[yNeta[10]] yEta10;
    bMat1_colshift += bK1_len[9];
    bMat2_colshift += bK2_len[9];
    for(i in 1:yK[10]){
      yXlt10[,i] = yX10[,i];
    }
    for(j in 1:yNeta[10]){
      yXlt10[j,lt_idx[10]] = yX10[j,lt_idx[10]] + lt[y10_Z1_id[j]];
    }
    yEta10 = evaluate_mu(evaluate_eta(yXlt10, y10_Z1, y10_Z2, y10_Z1_id, y10_Z2_id, yGamma10, yBeta10,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[10], y10_offset), 
                        family[10], link[10]);
    mean_PPD[10] = mean_PPD_rng(yEta10, yAux10, family[10]);
  }

  // Linear predictor for submodel 11
  if (M > 10) {
    vector[yNeta[11]] yEta11;
    bMat1_colshift += bK1_len[10];
    bMat2_colshift += bK2_len[10];
    for(i in 1:yK[11]){
      yXlt11[,i] = yX11[,i];
    }
    for(j in 1:yNeta[11]){
      yXlt11[j,lt_idx[11]] = yX11[j,lt_idx[11]] + lt[y11_Z1_id[j]];
    }
    yEta11 = evaluate_mu(evaluate_eta(yXlt11, y11_Z1, y11_Z2, y11_Z1_id, y11_Z2_id, yGamma11, yBeta11,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[11], y11_offset), 
                        family[11], link[11]);
    mean_PPD[11] = mean_PPD_rng(yEta11, yAux11, family[11]);
  }

  // Linear predictor for submodel 12
  if (M > 11) {
    vector[yNeta[12]] yEta12;
    bMat1_colshift += bK1_len[11];
    bMat2_colshift += bK2_len[11];
    for(i in 1:yK[12]){
      yXlt12[,i] = yX12[,i];
    }
    for(j in 1:yNeta[12]){
      yXlt12[j,lt_idx[12]] = yX12[j,lt_idx[12]] + lt[y12_Z1_id[j]];
    }
    yEta12 = evaluate_mu(evaluate_eta(yXlt12, y12_Z1, y12_Z2, y12_Z1_id, y12_Z2_id, yGamma12, yBeta12,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[12], y12_offset), 
                        family[12], link[12]);
    mean_PPD[12] = mean_PPD_rng(yEta12, yAux12, family[12]);
  }

  // Linear predictor for submodel 13
  if (M > 12) {
    vector[yNeta[13]] yEta13;
    bMat1_colshift += bK1_len[12];
    bMat2_colshift += bK2_len[12];
    for(i in 1:yK[13]){
      yXlt13[,i] = yX13[,i];
    }
    for(j in 1:yNeta[13]){
      yXlt13[j,lt_idx[13]] = yX13[j,lt_idx[13]] + lt[y13_Z1_id[j]];
    }
    yEta13 = evaluate_mu(evaluate_eta(yXlt13, y13_Z1, y13_Z2, y13_Z1_id, y13_Z2_id, yGamma13, yBeta13,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[13], y13_offset), 
                        family[13], link[13]);
    mean_PPD[13] = mean_PPD_rng(yEta13, yAux13, family[13]);
  }

  // Linear predictor for submodel 14
  if (M > 13) {
    vector[yNeta[14]] yEta14;
    bMat1_colshift += bK1_len[13];
    bMat2_colshift += bK2_len[13];
    for(i in 1:yK[14]){
      yXlt14[,i] = yX14[,i];
    }
    for(j in 1:yNeta[14]){
      yXlt14[j,lt_idx[14]] = yX14[j,lt_idx[14]] + lt[y14_Z1_id[j]];
    }
    yEta14 = evaluate_mu(evaluate_eta(yXlt14, y14_Z1, y14_Z2, y14_Z1_id, y14_Z2_id, yGamma14, yBeta14,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[14], y14_offset), 
                        family[14], link[14]);
    mean_PPD[14] = mean_PPD_rng(yEta14, yAux14, family[14]);
  }

  // Linear predictor for submodel 15
  if (M > 14) {
    vector[yNeta[15]] yEta15;
    bMat1_colshift += bK1_len[14];
    bMat2_colshift += bK2_len[14];
    for(i in 1:yK[15]){
      yXlt15[,i] = yX15[,i];
    }
    for(j in 1:yNeta[15]){
      yXlt15[j,lt_idx[15]] = yX15[j,lt_idx[15]] + lt[y15_Z1_id[j]];
    }
    yEta15 = evaluate_mu(evaluate_eta(yXlt15, y15_Z1, y15_Z2, y15_Z1_id, y15_Z2_id, yGamma15, yBeta15,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[15], y15_offset), 
                        family[15], link[15]);
    mean_PPD[15] = mean_PPD_rng(yEta15, yAux15, family[15]);
  }

  // Linear predictor for submodel 16
  if (M > 15) {
    vector[yNeta[16]] yEta16;
    bMat1_colshift += bK1_len[15];
    bMat2_colshift += bK2_len[15];
    for(i in 1:yK[16]){
      yXlt16[,i] = yX16[,i];
    }
    for(j in 1:yNeta[16]){
      yXlt16[j,lt_idx[16]] = yX16[j,lt_idx[16]] + lt[y16_Z1_id[j]];
    }
    yEta16 = evaluate_mu(evaluate_eta(yXlt16, y16_Z1, y16_Z2, y16_Z1_id, y16_Z2_id, yGamma16, yBeta16,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[16], y16_offset), 
                        family[16], link[16]);
    mean_PPD[16] = mean_PPD_rng(yEta16, yAux16, family[16]);
  }

  // Linear predictor for submodel 17
  if (M > 16) {
    vector[yNeta[17]] yEta17;
    bMat1_colshift += bK1_len[16];
    bMat2_colshift += bK2_len[16];
    for(i in 1:yK[17]){
      yXlt17[,i] = yX17[,i];
    }
    for(j in 1:yNeta[17]){
      yXlt17[j,lt_idx[17]] = yX17[j,lt_idx[17]] + lt[y17_Z1_id[j]];
    }
    yEta17 = evaluate_mu(evaluate_eta(yXlt17, y17_Z1, y17_Z2, y17_Z1_id, y17_Z2_id, yGamma17, yBeta17,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[17], y17_offset), 
                        family[17], link[17]);
    mean_PPD[17] = mean_PPD_rng(yEta17, yAux17, family[17]);
  }

  // Linear predictor for submodel 18
  if (M > 17) {
    vector[yNeta[18]] yEta18;
    bMat1_colshift += bK1_len[17];
    bMat2_colshift += bK2_len[17];
    for(i in 1:yK[18]){
      yXlt18[,i] = yX18[,i];
    }
    for(j in 1:yNeta[18]){
      yXlt18[j,lt_idx[18]] = yX18[j,lt_idx[18]] + lt[y18_Z1_id[j]];
    }
    yEta18 = evaluate_mu(evaluate_eta(yXlt18, y18_Z1, y18_Z2, y18_Z1_id, y18_Z2_id, yGamma18, yBeta18,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[18], y18_offset), 
                        family[18], link[18]);
    mean_PPD[18] = mean_PPD_rng(yEta18, yAux18, family[18]);
  }

  // Linear predictor for submodel 19
  if (M > 18) {
    vector[yNeta[19]] yEta19;
    bMat1_colshift += bK1_len[18];
    bMat2_colshift += bK2_len[18];
    for(i in 1:yK[19]){
      yXlt19[,i] = yX19[,i];
    }
    for(j in 1:yNeta[19]){
      yXlt19[j,lt_idx[19]] = yX19[j,lt_idx[19]] + lt[y19_Z1_id[j]];
    }
    yEta19 = evaluate_mu(evaluate_eta(yXlt19, y19_Z1, y19_Z2, y19_Z1_id, y19_Z2_id, yGamma19, yBeta19,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[19], y19_offset), 
                        family[19], link[19]);
    mean_PPD[19] = mean_PPD_rng(yEta19, yAux19, family[19]);
  }

  // Linear predictor for submodel 20
  if (M > 19) {
    vector[yNeta[20]] yEta20;
    bMat1_colshift += bK1_len[19];
    bMat2_colshift += bK2_len[19];
    for(i in 1:yK[20]){
      yXlt20[,i] = yX20[,i];
    }
    for(j in 1:yNeta[20]){
      yXlt20[j,lt_idx[20]] = yX20[j,lt_idx[20]] + lt[y20_Z1_id[j]];
    }
    yEta20 = evaluate_mu(evaluate_eta(yXlt20, y20_Z1, y20_Z2, y20_Z1_id, y20_Z2_id, yGamma20, yBeta20,
                                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[20], y20_offset), 
                        family[20], link[20]);
    mean_PPD[20] = mean_PPD_rng(yEta20, yAux20, family[20]);
  }
}

// Transform intercept parameters
if (intercept_type[1] > 0){
  for(i in 1:yK[1]){
    yXltbar1[i] = mean(yXlt1[,i]);
  }
  yAlpha1[1] = yGamma1[1] - dot_product(yXltbar1, yBeta1);
}
if (M > 1 && intercept_type[2] > 0){
  for(i in 1:yK[2]){
    yXltbar2[i] = mean(yXlt2[,i]);
  }
  yAlpha2[1] = yGamma2[1] - dot_product(yXltbar2, yBeta2);
}
if (M > 2 && intercept_type[3] > 0){
  for(i in 1:yK[3]){
    yXltbar3[i] = mean(yXlt3[,i]);
  }
  yAlpha3[1] = yGamma3[1] - dot_product(yXltbar3, yBeta3);
}
if (M > 3 && intercept_type[4] > 0){
  for(i in 1:yK[4]){
    yXltbar4[i] = mean(yXlt4[,i]);
  }
  yAlpha4[1] = yGamma4[1] - dot_product(yXltbar4, yBeta4);
}
if (M > 4 && intercept_type[5] > 0){
  for(i in 1:yK[5]){
    yXltbar5[i] = mean(yXlt5[,i]);
  }
  yAlpha5[1] = yGamma5[1] - dot_product(yXltbar5, yBeta5);
}
if (M > 5 && intercept_type[6] > 0){
  for(i in 1:yK[6]){
    yXltbar6[i] = mean(yXlt6[,i]);
  }
  yAlpha6[1] = yGamma6[1] - dot_product(yXltbar6, yBeta6);
}
if (M > 6 && intercept_type[7] > 0){
  for(i in 1:yK[7]){
    yXltbar7[i] = mean(yXlt7[,i]);
  }
  yAlpha7[1] = yGamma7[1] - dot_product(yXltbar7, yBeta7);
}
if (M > 7 && intercept_type[8] > 0){
  for(i in 1:yK[8]){
    yXltbar8[i] = mean(yXlt8[,i]);
  }
  yAlpha8[1] = yGamma8[1] - dot_product(yXltbar8, yBeta8);
}
if (M > 8 && intercept_type[9] > 0){
  for(i in 1:yK[9]){
    yXltbar9[i] = mean(yXlt9[,i]);
  }
  yAlpha9[1] = yGamma9[1] - dot_product(yXltbar9, yBeta9);
}
if (M > 9 && intercept_type[10] > 0){
  for(i in 1:yK[10]){
    yXltbar10[i] = mean(yXlt10[,i]);
  }
  yAlpha10[1] = yGamma10[1] - dot_product(yXltbar10, yBeta10);
}
if (M > 10 && intercept_type[11] > 0){
  for(i in 1:yK[11]){
    yXltbar11[i] = mean(yXlt11[,i]);
  }
  yAlpha11[1] = yGamma11[1] - dot_product(yXltbar11, yBeta11);
}
if (M > 11 && intercept_type[12] > 0){
  for(i in 1:yK[12]){
    yXltbar12[i] = mean(yXlt12[,i]);
  }
  yAlpha12[1] = yGamma12[1] - dot_product(yXltbar12, yBeta12);
}
if (M > 12 && intercept_type[13] > 0){
  for(i in 1:yK[13]){
    yXltbar13[i] = mean(yXlt13[,i]);
  }
  yAlpha13[1] = yGamma13[1] - dot_product(yXltbar13, yBeta13);
}
if (M > 13 && intercept_type[14] > 0){
  for(i in 1:yK[14]){
    yXltbar14[i] = mean(yXlt14[,i]);
  }
  yAlpha14[1] = yGamma14[1] - dot_product(yXltbar14, yBeta14);
}
if (M > 14 && intercept_type[15] > 0){
  for(i in 1:yK[15]){
    yXltbar15[i] = mean(yXlt15[,i]);
  }
  yAlpha15[1] = yGamma15[1] - dot_product(yXltbar15, yBeta15);
}
if (M > 15 && intercept_type[16] > 0){
  for(i in 1:yK[16]){
    yXltbar16[i] = mean(yXlt16[,i]);
  }
  yAlpha16[1] = yGamma16[1] - dot_product(yXltbar16, yBeta16);
}
if (M > 16 && intercept_type[17] > 0){
  for(i in 1:yK[17]){
    yXltbar17[i] = mean(yXlt17[,i]);
  }
  yAlpha17[1] = yGamma17[1] - dot_product(yXltbar17, yBeta17);
}
if (M > 17 && intercept_type[18] > 0){
  for(i in 1:yK[18]){
    yXltbar18[i] = mean(yXlt18[,i]);
  }
  yAlpha18[1] = yGamma18[1] - dot_product(yXltbar18, yBeta18);
}
if (M > 18 && intercept_type[19] > 0){
  for(i in 1:yK[19]){
    yXltbar19[i] = mean(yXlt19[,i]);
  }
  yAlpha19[1] = yGamma19[1] - dot_product(yXltbar19, yBeta19);
}
if (M > 19 && intercept_type[20] > 0){
  for(i in 1:yK[20]){
    yXltbar20[i] = mean(yXlt20[,i]);
  }
  yAlpha20[1] = yGamma20[1] - dot_product(yXltbar20, yBeta20);
}
  // Transform variance-covariance matrices

    // Grouping factor 1
  if (prior_dist_for_cov == 2 && bK1 == 1) {
    bCov1[1] = bSd1[1] * bSd1[1];
  }
  else if (prior_dist_for_cov == 2 && bK1 > 1) {
    bCov1 = to_vector(quad_form_diag(
      multiply_lower_tri_self_transpose(bCholesky1), bSd1))[bCov1_idx];
  }

  // Grouping factor 2
  if (prior_dist_for_cov == 2 && bK2 == 1) {
    bCov2[1] = bSd2[1] * bSd2[1];
  }
  else if (prior_dist_for_cov == 2 && bK2 > 1) {
    bCov2 = to_vector(quad_form_diag(
      multiply_lower_tri_self_transpose(bCholesky2), bSd2))[bCov2_idx];
  }
