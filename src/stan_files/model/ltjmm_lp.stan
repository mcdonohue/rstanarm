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

  vector[yNeta[1]] yEta1; // linear predictor
  vector[yNeta[2]] yEta2;
  vector[yNeta[3]] yEta3;
  vector[yNeta[4]] yEta4;
  vector[yNeta[5]] yEta5;
  vector[yNeta[6]] yEta6;
  vector[yNeta[7]] yEta7;
  vector[yNeta[8]] yEta8;
  vector[yNeta[9]] yEta9;
  vector[yNeta[10]] yEta10;
  vector[yNeta[11]] yEta11;
  vector[yNeta[12]] yEta12;
  vector[yNeta[13]] yEta13;
  vector[yNeta[14]] yEta14;
  vector[yNeta[15]] yEta15;
  vector[yNeta[16]] yEta16;
  vector[yNeta[17]] yEta17;
  vector[yNeta[18]] yEta18;
  vector[yNeta[19]] yEta19;
  vector[yNeta[20]] yEta20;
  

  aux_lp(sigma_lt_unscaled, y_prior_dist_for_sigma_lt,
         y_prior_scale_for_sigma_lt, y_prior_df_for_sigma_lt);
  target += normal_lpdf(lt | 0, sigma_lt);

  // Linear predictor for submodel 1
  if (M > 0) {
    int bMat1_colshift = 0; // column shift in bMat1
    int bMat2_colshift = 0; // column shift in bMat2
    for(i in 1:yK[1]){
      yXlt1[,i] = yX1[,i];
    }
    for(j in 1:yNeta[1]){
      yXlt1[j,lt_idx[1]] = yX1[j,lt_idx[1]] + lt[y1_Z1_id[j]];
    }
    yEta1 = evaluate_eta(yXlt1, y1_Z1, y1_Z2, y1_Z1_id, y1_Z2_id, yGamma1, yBeta1,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[1], y1_offset);
  }

  // Linear predictor for submodel 2
  if (M > 1) {
    int bMat1_colshift = bK1_len[1]; // column shift in bMat1
    int bMat2_colshift = bK2_len[1]; // column shift in bMat2
    for(i in 1:yK[2]){
      yXlt2[,i] = yX2[,i];
    }
    for(j in 1:yNeta[2]){
      yXlt2[j,lt_idx[2]] = yX2[j,lt_idx[2]] + lt[y2_Z1_id[j]];
    }
    yEta2 = evaluate_eta(yXlt2, y2_Z1, y2_Z2, y2_Z1_id, y2_Z2_id, yGamma2, yBeta2,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[2], y2_offset);
  }

  // Linear predictor for submodel 3
  if (M > 2) {
    int bMat1_colshift = sum(bK1_len[1:2]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:2]); // column shift in bMat2
    for(i in 1:yK[3]){
      yXlt3[,i] = yX3[,i];
    }
    for(j in 1:yNeta[3]){
      yXlt3[j,lt_idx[3]] = yX3[j,lt_idx[3]] + lt[y3_Z1_id[j]];
    }
    yEta3 = evaluate_eta(yXlt3, y3_Z1, y3_Z2, y3_Z1_id, y3_Z2_id, yGamma3, yBeta3,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[3], y3_offset);
  }

  // Linear predictor for submodel 4
  if (M > 3) {
    int bMat1_colshift = sum(bK1_len[1:3]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:3]); // column shift in bMat2
    for(i in 1:yK[4]){
      yXlt4[,i] = yX4[,i];
    }
    for(j in 1:yNeta[4]){
      yXlt4[j,lt_idx[4]] = yX4[j,lt_idx[4]] + lt[y4_Z1_id[j]];
    }
    yEta4 = evaluate_eta(yXlt4, y4_Z1, y4_Z2, y4_Z1_id, y4_Z2_id, yGamma4, yBeta4,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[4], y4_offset);
  }

  // Linear predictor for submodel 5
  if (M > 4) {
    int bMat1_colshift = sum(bK1_len[1:4]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:4]); // column shift in bMat2
    for(i in 1:yK[5]){
      yXlt5[,i] = yX5[,i];
    }
    for(j in 1:yNeta[5]){
      yXlt5[j,lt_idx[5]] = yX5[j,lt_idx[5]] + lt[y5_Z1_id[j]];
    }
    yEta5 = evaluate_eta(yXlt5, y5_Z1, y5_Z2, y5_Z1_id, y5_Z2_id, yGamma5, yBeta5,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[5], y5_offset);
  }

  // Linear predictor for submodel 6
  if (M > 5) {
    int bMat1_colshift = sum(bK1_len[1:5]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:5]); // column shift in bMat2
    for(i in 1:yK[6]){
      yXlt6[,i] = yX6[,i];
    }
    for(j in 1:yNeta[6]){
      yXlt6[j,lt_idx[6]] = yX6[j,lt_idx[6]] + lt[y6_Z1_id[j]];
    }
    yEta6 = evaluate_eta(yXlt6, y6_Z1, y6_Z2, y6_Z1_id, y6_Z2_id, yGamma6, yBeta6,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[6], y6_offset);
  }

  // Linear predictor for submodel 7
  if (M > 6) {
    int bMat1_colshift = sum(bK1_len[1:6]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:6]); // column shift in bMat2
    for(i in 1:yK[7]){
      yXlt7[,i] = yX7[,i];
    }
    for(j in 1:yNeta[7]){
      yXlt7[j,lt_idx[7]] = yX7[j,lt_idx[7]] + lt[y7_Z1_id[j]];
    }
    yEta7 = evaluate_eta(yXlt7, y7_Z1, y7_Z2, y7_Z1_id, y7_Z2_id, yGamma7, yBeta7,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[7], y7_offset);
  }

  // Linear predictor for submodel 8
  if (M > 7) {
    int bMat1_colshift = sum(bK1_len[1:7]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:7]); // column shift in bMat2
    for(i in 1:yK[8]){
      yXlt8[,i] = yX8[,i];
    }
    for(j in 1:yNeta[8]){
      yXlt8[j,lt_idx[8]] = yX8[j,lt_idx[8]] + lt[y8_Z1_id[j]];
    }
    yEta8 = evaluate_eta(yXlt8, y8_Z1, y8_Z2, y8_Z1_id, y8_Z2_id, yGamma8, yBeta8,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[8], y8_offset);
  }

  // Linear predictor for submodel 9
  if (M > 8) {
    int bMat1_colshift = sum(bK1_len[1:8]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:8]); // column shift in bMat2
    for(i in 1:yK[9]){
      yXlt9[,i] = yX9[,i];
    }
    for(j in 1:yNeta[9]){
      yXlt9[j,lt_idx[9]] = yX9[j,lt_idx[9]] + lt[y9_Z1_id[j]];
    }
    yEta9 = evaluate_eta(yXlt9, y9_Z1, y9_Z2, y9_Z1_id, y9_Z2_id, yGamma9, yBeta9,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[9], y9_offset);
  }

  // Linear predictor for submodel 10
  if (M > 9) {
    int bMat1_colshift = sum(bK1_len[1:9]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:9]); // column shift in bMat2
    for(i in 1:yK[10]){
      yXlt10[,i] = yX10[,i];
    }
    for(j in 1:yNeta[10]){
      yXlt10[j,lt_idx[10]] = yX10[j,lt_idx[10]] + lt[y10_Z1_id[j]];
    }
    yEta10 = evaluate_eta(yXlt10, y10_Z1, y10_Z2, y10_Z1_id, y10_Z2_id, yGamma10, yBeta10,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[10], y10_offset);
  }

  // Linear predictor for submodel 11
  if (M > 10) {
    int bMat1_colshift = sum(bK1_len[1:10]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:10]); // column shift in bMat2
    for(i in 1:yK[11]){
      yXlt11[,i] = yX11[,i];
    }
    for(j in 1:yNeta[11]){
      yXlt11[j,lt_idx[11]] = yX11[j,lt_idx[11]] + lt[y11_Z1_id[j]];
    }
    yEta11 = evaluate_eta(yXlt11, y11_Z1, y11_Z2, y11_Z1_id, y11_Z2_id, yGamma11, yBeta11,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[11], y11_offset);
  }

  // Linear predictor for submodel 12
  if (M > 11) {
    int bMat1_colshift = sum(bK1_len[1:11]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:11]); // column shift in bMat2
    for(i in 1:yK[12]){
      yXlt12[,i] = yX12[,i];
    }
    for(j in 1:yNeta[12]){
      yXlt12[j,lt_idx[12]] = yX12[j,lt_idx[12]] + lt[y12_Z1_id[j]];
    }
    yEta12 = evaluate_eta(yXlt12, y12_Z1, y12_Z2, y12_Z1_id, y12_Z2_id, yGamma12, yBeta12,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[12], y12_offset);
  }

  // Linear predictor for submodel 13
  if (M > 12) {
    int bMat1_colshift = sum(bK1_len[1:12]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:12]); // column shift in bMat2
    for(i in 1:yK[13]){
      yXlt13[,i] = yX13[,i];
    }
    for(j in 1:yNeta[13]){
      yXlt13[j,lt_idx[13]] = yX13[j,lt_idx[13]] + lt[y13_Z1_id[j]];
    }
    yEta13 = evaluate_eta(yXlt13, y13_Z1, y13_Z2, y13_Z1_id, y13_Z2_id, yGamma13, yBeta13,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[13], y13_offset);
  }

  // Linear predictor for submodel 14
  if (M > 13) {
    int bMat1_colshift = sum(bK1_len[1:13]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:13]); // column shift in bMat2
    for(i in 1:yK[14]){
      yXlt14[,i] = yX14[,i];
    }
    for(j in 1:yNeta[14]){
      yXlt14[j,lt_idx[14]] = yX14[j,lt_idx[14]] + lt[y14_Z1_id[j]];
    }
    yEta14 = evaluate_eta(yXlt14, y14_Z1, y14_Z2, y14_Z1_id, y14_Z2_id, yGamma14, yBeta14,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[14], y14_offset);
  }

  // Linear predictor for submodel 15
  if (M > 14) {
    int bMat1_colshift = sum(bK1_len[1:14]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:14]); // column shift in bMat2
    for(i in 1:yK[15]){
      yXlt15[,i] = yX15[,i];
    }
    for(j in 1:yNeta[15]){
      yXlt15[j,lt_idx[15]] = yX15[j,lt_idx[15]] + lt[y15_Z1_id[j]];
    }
    yEta15 = evaluate_eta(yXlt15, y15_Z1, y15_Z2, y15_Z1_id, y15_Z2_id, yGamma15, yBeta15,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[15], y15_offset);
  }

  // Linear predictor for submodel 16
  if (M > 15) {
    int bMat1_colshift = sum(bK1_len[1:15]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:15]); // column shift in bMat2
    for(i in 1:yK[16]){
      yXlt16[,i] = yX16[,i];
    }
    for(j in 1:yNeta[16]){
      yXlt16[j,lt_idx[16]] = yX16[j,lt_idx[16]] + lt[y16_Z1_id[j]];
    }
    yEta16 = evaluate_eta(yXlt16, y16_Z1, y16_Z2, y16_Z1_id, y16_Z2_id, yGamma16, yBeta16,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[16], y16_offset);
  }

  // Linear predictor for submodel 17
  if (M > 16) {
    int bMat1_colshift = sum(bK1_len[1:16]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:16]); // column shift in bMat2
    for(i in 1:yK[17]){
      yXlt17[,i] = yX17[,i];
    }
    for(j in 1:yNeta[17]){
      yXlt17[j,lt_idx[17]] = yX17[j,lt_idx[17]] + lt[y17_Z1_id[j]];
    }
    yEta17 = evaluate_eta(yXlt17, y17_Z1, y17_Z2, y17_Z1_id, y17_Z2_id, yGamma17, yBeta17,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[17], y17_offset);
  }

  // Linear predictor for submodel 18
  if (M > 17) {
    int bMat1_colshift = sum(bK1_len[1:17]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:17]); // column shift in bMat2
    for(i in 1:yK[18]){
      yXlt18[,i] = yX18[,i];
    }
    for(j in 1:yNeta[18]){
      yXlt18[j,lt_idx[18]] = yX18[j,lt_idx[18]] + lt[y18_Z1_id[j]];
    }
    yEta18 = evaluate_eta(yXlt18, y18_Z1, y18_Z2, y18_Z1_id, y18_Z2_id, yGamma18, yBeta18,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[18], y18_offset);
  }

  // Linear predictor for submodel 19
  if (M > 18) {
    int bMat1_colshift = sum(bK1_len[1:18]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:18]); // column shift in bMat2
    for(i in 1:yK[19]){
      yXlt19[,i] = yX19[,i];
    }
    for(j in 1:yNeta[19]){
      yXlt19[j,lt_idx[19]] = yX19[j,lt_idx[19]] + lt[y19_Z1_id[j]];
    }
    yEta19 = evaluate_eta(yXlt19, y19_Z1, y19_Z2, y19_Z1_id, y19_Z2_id, yGamma19, yBeta19,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[19], y19_offset);
  }

  // Linear predictor for submodel 20
  if (M > 19) {
    int bMat1_colshift = sum(bK1_len[1:19]); // column shift in bMat1
    int bMat2_colshift = sum(bK2_len[1:19]); // column shift in bMat2
    for(i in 1:yK[20]){
      yXlt20[,i] = yX20[,i];
    }
    for(j in 1:yNeta[20]){
      yXlt20[j,lt_idx[20]] = yX20[j,lt_idx[20]] + lt[y20_Z1_id[j]];
    }
    yEta20 = evaluate_eta(yXlt20, y20_Z1, y20_Z2, y20_Z1_id, y20_Z2_id, yGamma20, yBeta20,
                         bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[20], y20_offset);
  }  
  

  // Log-likelihoods
  if (prior_PD == 0) {
    glm_lp(yReal1, yInt1, yEta1, yAux1, family[1], link[1], sum_log_y1, sqrt_y1, log_y1, weights_y1);
    if (M > 1)
      glm_lp(yReal2, yInt2, yEta2, yAux2, family[2], link[2], sum_log_y2, sqrt_y2, log_y2, weights_y2);
    if (M > 2)
      glm_lp(yReal3, yInt3, yEta3, yAux3, family[3], link[3], sum_log_y3, sqrt_y3, log_y3, weights_y3);
    if (M > 3)
      glm_lp(yReal4, yInt4, yEta4, yAux4, family[4], link[4], sum_log_y4, sqrt_y4, log_y4, weights_y4);
    if (M > 4)
      glm_lp(yReal5, yInt5, yEta5, yAux5, family[5], link[5], sum_log_y5, sqrt_y5, log_y5, weights_y5);
    if (M > 5)
      glm_lp(yReal6, yInt6, yEta6, yAux6, family[6], link[6], sum_log_y6, sqrt_y6, log_y6, weights_y6);
    if (M > 6)
      glm_lp(yReal7, yInt7, yEta7, yAux7, family[7], link[7], sum_log_y7, sqrt_y7, log_y7, weights_y7);
    if (M > 7)
      glm_lp(yReal8, yInt8, yEta8, yAux8, family[8], link[8], sum_log_y8, sqrt_y8, log_y8, weights_y8);
    if (M > 8)
      glm_lp(yReal9, yInt9, yEta9, yAux9, family[9], link[9], sum_log_y9, sqrt_y9, log_y9, weights_y9);
    if (M > 9)
      glm_lp(yReal10, yInt10, yEta10, yAux10, family[10], link[10], sum_log_y10, sqrt_y10, log_y10, weights_y10);
    if (M > 10)
      glm_lp(yReal11, yInt11, yEta11, yAux11, family[11], link[11], sum_log_y11, sqrt_y11, log_y11, weights_y11);
    if (M > 11)
      glm_lp(yReal12, yInt12, yEta12, yAux12, family[12], link[12], sum_log_y12, sqrt_y12, log_y12, weights_y12);
    if (M > 12)
      glm_lp(yReal13, yInt13, yEta13, yAux13, family[13], link[13], sum_log_y13, sqrt_y13, log_y13, weights_y13);
    if (M > 13)
      glm_lp(yReal14, yInt14, yEta14, yAux14, family[14], link[14], sum_log_y14, sqrt_y14, log_y14, weights_y14);
    if (M > 14)
      glm_lp(yReal15, yInt15, yEta15, yAux15, family[15], link[15], sum_log_y15, sqrt_y15, log_y15, weights_y15);
    if (M > 15)
      glm_lp(yReal16, yInt16, yEta16, yAux16, family[16], link[16], sum_log_y16, sqrt_y16, log_y16, weights_y16);
    if (M > 16)
      glm_lp(yReal17, yInt17, yEta17, yAux17, family[17], link[17], sum_log_y17, sqrt_y17, log_y17, weights_y17);
    if (M > 17)
      glm_lp(yReal18, yInt18, yEta18, yAux18, family[18], link[18], sum_log_y18, sqrt_y18, log_y18, weights_y18);
    if (M > 18)
      glm_lp(yReal19, yInt19, yEta19, yAux19, family[19], link[19], sum_log_y19, sqrt_y19, log_y19, weights_y19);
    if (M > 19)
      glm_lp(yReal20, yInt20, yEta20, yAux20, family[20], link[20], sum_log_y20, sqrt_y20, log_y20, weights_y20);  
  }
