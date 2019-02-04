from src.hic.blocks_parser import parse_to_df, create_bg
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator, \
    CmBFunctionGammaEstimator, get_d_from_cycles, get_b_from_cycles, GammaEstimator
from src.real_data import without_c1

folder = "real_data/EUA/300Kbp"
sp1 = "APCF"
sp2 = "hg19"
file_name = "%s/%s_%s.merged.map" % (folder, sp1, sp2) # if APCF
# file_name = "real_data/BOR/300Kbp/SFs/SFs_%s/Orthology.Blocks" % sp2

df = parse_to_df(file_name)
g = create_bg(df, sp1, sp2, cyclic=False)
g_with_miss_edges = create_bg(df, sp1, sp2, cyclic=False, add_missing_edges=True)
c = without_c1(g_with_miss_edges.count_cycles())

d = g.d()
b = g.b()

dir_est = DirichletEstimator()
uni_est = UniformEstimator()
fcms = 7
first_cms = FirstCmsDirEstimator(fcms)
gmm = 0.15
gmm2 = 0.3
gamma_est = GammaEstimator(gmm, 50)
gamma_est2 = GammaEstimator(0.3, 50)
print(get_d_from_cycles(c), d, g.p_odd(), g.p_even())

print(b)
print(d/b)
print("min: %d\tunif: %.2f\tdir: %.2f\tfirst_cms_dir_%d: %.2f\tgamma_%.2f: %.2f\tgamma_%.2f: %.2f" %
      (d,
       uni_est.predict_k_by_db(d, b),
       dir_est.predict_k_by_db(d, b),
       fcms,
       first_cms.predict_k(c),
       gmm,
       gamma_est.predict_k_by_db(d, b),
       gmm2,
       gamma_est2.predict_k_by_db(d, b)))
