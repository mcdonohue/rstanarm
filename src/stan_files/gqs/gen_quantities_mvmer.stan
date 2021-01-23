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

  // Evaluate mean_PPD
  {
    int bMat1_colshift = 0; // column shift in bMat1
    int bMat2_colshift = 0; // column shift in bMat2

    // Linear predictor for submodel 1
    if (M > 0) {
      vector[yNeta[1]] yEta1 = evaluate_mu( // linear predictor
        evaluate_eta(yX1, y1_Z1, y1_Z2, y1_Z1_id, y1_Z2_id, yGamma1, yBeta1,
                     bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[1], 
                     y1_offset),
        family[1], link[1]);
      mean_PPD[1] = mean_PPD_rng(yEta1, yAux1, family[1]);
    }

    // Linear predictor for submodel 2
    if (M > 1) {
      vector[yNeta[2]] yEta2;
      bMat1_colshift += bK1_len[1];
      bMat2_colshift += bK2_len[1];
      yEta2 = evaluate_mu(evaluate_eta(yX2, y2_Z1, y2_Z2, y2_Z1_id, y2_Z2_id, yGamma2, yBeta2,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[2], 
                                       y2_offset), 
                          family[2], link[2]);
      mean_PPD[2] = mean_PPD_rng(yEta2, yAux2, family[2]);
    }
    
    // Linear predictor for submodel 3
    if (M > 2) {
      vector[yNeta[3]] yEta3;
      bMat1_colshift += bK1_len[2];
      bMat2_colshift += bK2_len[2];
      yEta3 = evaluate_mu(evaluate_eta(yX3, y3_Z1, y3_Z2, y3_Z1_id, y3_Z2_id, yGamma3, yBeta3,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[3], 
                                       y3_offset), 
                          family[3], link[3]);
      mean_PPD[3] = mean_PPD_rng(yEta3, yAux3, family[3]);
    }

   // Linear predictor for submodel 4
    if (M > 3) {
      vector[yNeta[4]] yEta4;
      bMat1_colshift += bK1_len[3];
      bMat2_colshift += bK2_len[3];
      yEta4 = evaluate_mu(evaluate_eta(yX4, y4_Z1, y4_Z2, y4_Z1_id, y4_Z2_id, yGamma4, yBeta4,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[4],
                                       y4_offset), 
                          family[4], link[4]);
      mean_PPD[4] = mean_PPD_rng(yEta4, yAux4, family[4]);
    }

    // Linear predictor for submodel 5
    if (M > 4) {
      vector[yNeta[5]] yEta5;
      bMat1_colshift += bK1_len[4];
      bMat2_colshift += bK2_len[4];
      yEta5 = evaluate_mu(evaluate_eta(yX5, y5_Z1, y5_Z2, y5_Z1_id, y5_Z2_id, yGamma5, yBeta5,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[5],
                                       y5_offset), 
                          family[5], link[5]);
      mean_PPD[5] = mean_PPD_rng(yEta5, yAux5, family[5]);
    }

    // Linear predictor for submodel 6
    if (M > 5) {
      vector[yNeta[6]] yEta6;
      bMat1_colshift += bK1_len[5];
      bMat2_colshift += bK2_len[5];
      yEta6 = evaluate_mu(evaluate_eta(yX6, y6_Z1, y6_Z2, y6_Z1_id, y6_Z2_id, yGamma6, yBeta6,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[6],
                                       y6_offset), 
                          family[6], link[6]);
      mean_PPD[6] = mean_PPD_rng(yEta6, yAux6, family[6]);
    }

    // Linear predictor for submodel 7
    if (M > 6) {
      vector[yNeta[7]] yEta7;
      bMat1_colshift += bK1_len[6];
      bMat2_colshift += bK2_len[6];
      yEta7 = evaluate_mu(evaluate_eta(yX7, y7_Z1, y7_Z2, y7_Z1_id, y7_Z2_id, yGamma7, yBeta7,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[7],
                                       y7_offset), 
                          family[7], link[7]);
      mean_PPD[7] = mean_PPD_rng(yEta7, yAux7, family[7]);
    }

    // Linear predictor for submodel 8
    if (M > 7) {
      vector[yNeta[8]] yEta8;
      bMat1_colshift += bK1_len[7];
      bMat2_colshift += bK2_len[7];
      yEta8 = evaluate_mu(evaluate_eta(yX8, y8_Z1, y8_Z2, y8_Z1_id, y8_Z2_id, yGamma8, yBeta8,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[8],
                                       y8_offset), 
                          family[8], link[8]);
      mean_PPD[8] = mean_PPD_rng(yEta8, yAux8, family[8]);
    }

    // Linear predictor for submodel 9
    if (M > 8) {
      vector[yNeta[9]] yEta9;
      bMat1_colshift += bK1_len[8];
      bMat2_colshift += bK2_len[8];
      yEta9 = evaluate_mu(evaluate_eta(yX9, y9_Z1, y9_Z2, y9_Z1_id, y9_Z2_id, yGamma9, yBeta9,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[9],
                                       y9_offset), 
                          family[9], link[9]);
      mean_PPD[9] = mean_PPD_rng(yEta9, yAux9, family[9]);
    }

    // Linear predictor for submodel 10
    if (M > 9) {
      vector[yNeta[10]] yEta10;
      bMat1_colshift += bK1_len[9];
      bMat2_colshift += bK2_len[9];
      yEta10 = evaluate_mu(evaluate_eta(yX10, y10_Z1, y10_Z2, y10_Z1_id, y10_Z2_id, yGamma10, yBeta10,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[10],
                                       y10_offset), 
                          family[10], link[10]);
      mean_PPD[10] = mean_PPD_rng(yEta10, yAux10, family[10]);
    }

    // Linear predictor for submodel 11
    if (M > 10) {
      vector[yNeta[11]] yEta11;
      bMat1_colshift += bK1_len[10];
      bMat2_colshift += bK2_len[10];
      yEta11 = evaluate_mu(evaluate_eta(yX11, y11_Z1, y11_Z2, y11_Z1_id, y11_Z2_id, yGamma11, yBeta11,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[11],
                                       y11_offset), 
                          family[11], link[11]);
      mean_PPD[11] = mean_PPD_rng(yEta11, yAux11, family[11]);
    }

    // Linear predictor for submodel 12
    if (M > 11) {
      vector[yNeta[12]] yEta12;
      bMat1_colshift += bK1_len[11];
      bMat2_colshift += bK2_len[11];
      yEta12 = evaluate_mu(evaluate_eta(yX12, y12_Z1, y12_Z2, y12_Z1_id, y12_Z2_id, yGamma12, yBeta12,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[12],
                                       y12_offset), 
                          family[12], link[12]);
      mean_PPD[12] = mean_PPD_rng(yEta12, yAux12, family[12]);
    }

    // Linear predictor for submodel 13
    if (M > 12) {
      vector[yNeta[13]] yEta13;
      bMat1_colshift += bK1_len[12];
      bMat2_colshift += bK2_len[12];
      yEta13 = evaluate_mu(evaluate_eta(yX13, y13_Z1, y13_Z2, y13_Z1_id, y13_Z2_id, yGamma13, yBeta13,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[13],
                                       y13_offset), 
                          family[13], link[13]);
      mean_PPD[13] = mean_PPD_rng(yEta13, yAux13, family[13]);
    }

    // Linear predictor for submodel 14
    if (M > 13) {
      vector[yNeta[14]] yEta14;
      bMat1_colshift += bK1_len[13];
      bMat2_colshift += bK2_len[13];
      yEta14 = evaluate_mu(evaluate_eta(yX14, y14_Z1, y14_Z2, y14_Z1_id, y14_Z2_id, yGamma14, yBeta14,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[14],
                                       y14_offset), 
                          family[14], link[14]);
      mean_PPD[14] = mean_PPD_rng(yEta14, yAux14, family[14]);
    }

    // Linear predictor for submodel 15
    if (M > 14) {
      vector[yNeta[15]] yEta15;
      bMat1_colshift += bK1_len[14];
      bMat2_colshift += bK2_len[14];
      yEta15 = evaluate_mu(evaluate_eta(yX15, y15_Z1, y15_Z2, y15_Z1_id, y15_Z2_id, yGamma15, yBeta15,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[15],
                                       y15_offset), 
                          family[15], link[15]);
      mean_PPD[15] = mean_PPD_rng(yEta15, yAux15, family[15]);
    }

    // Linear predictor for submodel 16
    if (M > 15) {
      vector[yNeta[16]] yEta16;
      bMat1_colshift += bK1_len[15];
      bMat2_colshift += bK2_len[15];
      yEta16 = evaluate_mu(evaluate_eta(yX16, y16_Z1, y16_Z2, y16_Z1_id, y16_Z2_id, yGamma16, yBeta16,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[16],
                                       y16_offset), 
                          family[16], link[16]);
      mean_PPD[16] = mean_PPD_rng(yEta16, yAux16, family[16]);
    }

    // Linear predictor for submodel 17
    if (M > 16) {
      vector[yNeta[17]] yEta17;
      bMat1_colshift += bK1_len[16];
      bMat2_colshift += bK2_len[16];
      yEta17 = evaluate_mu(evaluate_eta(yX17, y17_Z1, y17_Z2, y17_Z1_id, y17_Z2_id, yGamma17, yBeta17,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[17],
                                       y17_offset), 
                          family[17], link[17]);
      mean_PPD[17] = mean_PPD_rng(yEta17, yAux17, family[17]);
    }

    // Linear predictor for submodel 18
    if (M > 17) {
      vector[yNeta[18]] yEta18;
      bMat1_colshift += bK1_len[17];
      bMat2_colshift += bK2_len[17];
      yEta18 = evaluate_mu(evaluate_eta(yX18, y18_Z1, y18_Z2, y18_Z1_id, y18_Z2_id, yGamma18, yBeta18,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[18],
                                       y18_offset), 
                          family[18], link[18]);
      mean_PPD[18] = mean_PPD_rng(yEta18, yAux18, family[18]);
    }

    // Linear predictor for submodel 19
    if (M > 18) {
      vector[yNeta[19]] yEta19;
      bMat1_colshift += bK1_len[18];
      bMat2_colshift += bK2_len[18];
      yEta19 = evaluate_mu(evaluate_eta(yX19, y19_Z1, y19_Z2, y19_Z1_id, y19_Z2_id, yGamma19, yBeta19,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[19],
                                       y19_offset), 
                          family[19], link[19]);
      mean_PPD[19] = mean_PPD_rng(yEta19, yAux19, family[19]);
    }

    // Linear predictor for submodel 20
    if (M > 19) {
      vector[yNeta[20]] yEta20;
      bMat1_colshift += bK1_len[19];
      bMat2_colshift += bK2_len[19];
      yEta20 = evaluate_mu(evaluate_eta(yX20, y20_Z1, y20_Z2, y20_Z1_id, y20_Z2_id, yGamma20, yBeta20,
                                       bMat1, bMat2, bMat1_colshift, bMat2_colshift, intercept_type[20],
                                       y20_offset), 
                          family[20], link[20]);
      mean_PPD[20] = mean_PPD_rng(yEta20, yAux20, family[20]);
    }
  }

  // Transform intercept parameters
    if (intercept_type[1] > 0)
    yAlpha1[1] = yGamma1[1] - dot_product(yXbar1, yBeta1);
  if (M > 1 && intercept_type[2] > 0)
    yAlpha2[1] = yGamma2[1] - dot_product(yXbar2, yBeta2);
  if (M > 2 && intercept_type[3] > 0)
    yAlpha3[1] = yGamma3[1] - dot_product(yXbar3, yBeta3);
  if (M > 3 && intercept_type[4] > 0)
    yAlpha4[1] = yGamma4[1] - dot_product(yXbar4, yBeta4);
  if (M > 4 && intercept_type[5] > 0)
    yAlpha5[1] = yGamma5[1] - dot_product(yXbar5, yBeta5);
  if (M > 5 && intercept_type[6] > 0)
    yAlpha6[1] = yGamma6[1] - dot_product(yXbar6, yBeta6);
  if (M > 6 && intercept_type[7] > 0)
    yAlpha7[1] = yGamma7[1] - dot_product(yXbar7, yBeta7);
  if (M > 7 && intercept_type[8] > 0)
    yAlpha8[1] = yGamma8[1] - dot_product(yXbar8, yBeta8);
  if (M > 8 && intercept_type[9] > 0)
    yAlpha9[1] = yGamma9[1] - dot_product(yXbar9, yBeta9);
  if (M > 9 && intercept_type[10] > 0)
    yAlpha10[1] = yGamma10[1] - dot_product(yXbar10, yBeta10);
  if (M > 10 && intercept_type[11] > 0)
    yAlpha11[1] = yGamma11[1] - dot_product(yXbar11, yBeta11);
  if (M > 11 && intercept_type[12] > 0)
    yAlpha12[1] = yGamma12[1] - dot_product(yXbar12, yBeta12);
  if (M > 12 && intercept_type[13] > 0)
    yAlpha13[1] = yGamma13[1] - dot_product(yXbar13, yBeta13);
  if (M > 13 && intercept_type[14] > 0)
    yAlpha14[1] = yGamma14[1] - dot_product(yXbar14, yBeta14);
  if (M > 14 && intercept_type[15] > 0)
    yAlpha15[1] = yGamma15[1] - dot_product(yXbar15, yBeta15);
  if (M > 15 && intercept_type[16] > 0)
    yAlpha16[1] = yGamma16[1] - dot_product(yXbar16, yBeta16);
  if (M > 16 && intercept_type[17] > 0)
    yAlpha17[1] = yGamma17[1] - dot_product(yXbar17, yBeta17);
  if (M > 17 && intercept_type[18] > 0)
    yAlpha18[1] = yGamma18[1] - dot_product(yXbar18, yBeta18);
  if (M > 18 && intercept_type[19] > 0)
    yAlpha19[1] = yGamma19[1] - dot_product(yXbar19, yBeta19);
  if (M > 19 && intercept_type[20] > 0)
    yAlpha20[1] = yGamma20[1] - dot_product(yXbar20, yBeta20);    

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
