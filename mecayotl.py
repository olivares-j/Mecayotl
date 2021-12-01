import sys
import os
import numpy as np
import pandas as pd
import h5py

from scipy.stats import multivariate_normal
from astropy.table import Table
from pygaia.astrometry.vectorastrometry import phase_space_to_astrometry
from pygaia.astrometry.constants import au_km_year_per_sec,au_mas_parsec

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from matplotlib import lines as mlines
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize,TwoSlopeNorm
from tqdm import tqdm

def get_principal(sigma,level=2.0):
    sd_x   = np.sqrt(sigma[0,0])
    sd_y   = np.sqrt(sigma[1,1])
    rho_xy = sigma[0,1]/(sd_x*sd_y)


    # Author: Jake VanderPlas
    # License: BSD
    #----------------------------------------
    sigma_xy2 = rho_xy * sd_x * sd_y

    alpha = 0.5 * np.arctan2(2 * sigma_xy2,(sd_x ** 2 - sd_y ** 2))
    tmp1  = 0.5 * (sd_x ** 2 + sd_y ** 2)
    tmp2  = np.sqrt(0.25 * (sd_x ** 2 - sd_y ** 2) ** 2 + sigma_xy2 ** 2)

    return level*np.sqrt(tmp1 + tmp2), level*np.sqrt(np.abs(tmp1 - tmp2)), alpha* 180. / np.pi

class Mecayotl(object):
	"""
	Mecayotl is an algorithm desgined to identify members of open clusters.
	Mecayotl means: genealogia o parentesco.
	https://gdn.iib.unam.mx/termino/search?queryCreiterio=mecayotl&queryPartePalabra=cualquiera&queryBuscarEn=nahuatl&queryLimiteRegistros=50 
	"""

	def __init__(self,
					path_main,
					path_mcmichael = "/home/jolivares/Repos/McMichael/",
					path_amasijo   = "/home/jolivares/Repos/Amasijo/",
					cmap_probability="coolwarm",
					cmap_features="viridis_r",
					seed=1234):

		#------ Set Seed -----------------
		np.random.seed(seed=seed)
		self.random_state = np.random.RandomState(seed=seed)
		#----------------------------------------------------

		#------------- Directories -----------------
		self.dir_mcmi      = path_mcmichael
		self.dir_amasijo   = path_amasijo
		self.path_main     = path_main
		#--------------------------------------

		#---------------- Files -----------------
		self.file_samples   = self.path_main + "Real/Data/members_synthetic.csv"

		self.cmap_prob = plt.get_cmap(cmap_probability)
		self.cmap_feat = plt.get_cmap(cmap_features)
		self.idxs      = [[0,1],[2,1],[0,2],[3,4],[5,4],[3,5]]
		self.plots     = [["ra","dec"],["pmra","pmdec"],["parallax","pmdec"],["g_rp","g"]]
		
		self.labels_obs = ["ra","dec","parallax","pmra","pmdec","radial_velocity"]
		gaia_observables = [
		"ra","dec","parallax","pmra","pmdec","dr2_radial_velocity",
		"ra_error","dec_error","parallax_error","pmra_error","pmdec_error","dr2_radial_velocity_error",
		"ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
		"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
		"parallax_pmra_corr","parallax_pmdec_corr",
		"pmra_pmdec_corr"]
		self.OBS = gaia_observables[0:6]
		self.UNC = gaia_observables[6:12]
		self.RHO = gaia_observables[12:]

		#------------- Cases -------------------------------
		self.file_hdf_base   = path_main + "/{0}/Data/catalogue.h5"
		self.file_data_base  = path_main + "/{0}/Data/data.h5"
		self.file_model_base = path_main + "/{0}/Models/{1}_GMM_{2}.h5"
		self.file_comparison = path_main + "/{0}/Models/{1}_comparison.png"
		self.cases = [
		{"name":"Field",
		"data_name":"mu_field",
		"components":range(10,15,1),
		"file_base": "{0}/Field_GMM_{1}.h5",
		"file_comparison": "{0}/Field_comparison.png",
		"file_plot": "{0}/Field_GMM_best.pdf"},
		{"name":"Cluster",
		"data_name":"mu_syn",
		"components":range(1,11,1),
		"file_base": "{0}/Cluster_GMM_{1}.h5",
		"file_comparison":"{0}/Cluster_comparison.png",
		"file_plot":"{0}/Cluster_GMM_best.pdf"}
		]
		#-------------------------------------------------- 

	def _initialize_mcmichael(self):
		#======================= Libraries ===============================================
		#-------------- Commands to replace dimension -----------------------------
		cmd = 'sed -e "s|DIMENSION|{0}|g"'.format(6)
		cmd += ' {0}GPU/Functions_base.py > {0}GPU/Functions.py'.format(self.dir_mcmi)
		os.system(cmd)
		#--------------------------------------------------------------------------
		from GPU.gmm import GaussianMixture
		#=================================================================================

	def _initialize_amasijo(self):
		sys.path.append(self.dir_amasijo)
		from Amasijo import Amasijo
		from Quality import ClassifierQuality


	def generate_true_cluster(self,file_kalkayotl,n_samples=100):
		"""
		Generate synthetic data based on Kalkayotl input parameters
		"""
		param = pd.read_csv(file_kalkayotl,usecols=["Parameter","mode"])

		#---- Extract parameters ------------------------------------------------
		loc  = param.loc[param["Parameter"].str.contains("loc"),"mode"].values
		param.fillna(value=1.0,inplace=True)
		stds = param.loc[param["Parameter"].str.contains('stds'),"mode"].values
		corr = param.loc[param["Parameter"].str.contains('corr'),"mode"].values
		#------------------------------------------------------------------------

		#---- Construct covariance --------------
		stds = np.diag(stds)
		corr = np.reshape(corr,(6,6))
		cov  = np.dot(stds,corr.dot(stds))
		#-----------------------------------------

		#--------- Generate synthetic samples ----------------------
		phase_space = multivariate_normal(mean=loc,cov=cov).rvs(
						size=n_samples,
						random_state=self.random_state)
		astrometry_rv  = np.array(phase_space_to_astrometry(
						phase_space[:,0],
						phase_space[:,1],
						phase_space[:,2],
						phase_space[:,3],
						phase_space[:,4],
						phase_space[:,5]
						))
		astrometry_rv[0] = np.rad2deg(astrometry_rv[0])
		astrometry_rv[1] = np.rad2deg(astrometry_rv[1])
		#------------------------------------------------------------

		#---------- Data Frame original -------------------
		df = pd.DataFrame(data={
						"ra":astrometry_rv[0],
						"dec":astrometry_rv[1],
						"parallax":astrometry_rv[2],
						"pmra":astrometry_rv[3],
						"pmdec":astrometry_rv[4],
						"dr2_radial_velocity":astrometry_rv[5]
						})
		df.to_csv(self.file_samples,index_label="source_id")
		#---------------------------------------------------


	def assemble_data(self,file_catalogue,file_members,
						n_fld=100000,instance="Real"):
		#------------ Files ------------------------------
		file_data = self.file_data_base.format(instance)
		file_hdf  = self.file_hdf_base.format(instance)
		#-------------------------------------------------

		#--------------- Read ------------------------------
		cat = Table.read(file_catalogue, format='fits')
		df_cat = cat.to_pandas()
		df_mem = pd.read_csv(file_members,usecols=["source_id"])
		df_syn = pd.read_csv(self.file_samples,usecols=self.OBS)
		#---------------------------------------------------

		n_sources = df_cat.shape[0]
		mu_syn    = df_syn.to_numpy()

		#----------- Covariance matrices ------------------
		print("Filling covariance matrices ...")
		sg_data = np.zeros((n_sources,6,6))

		#------ Extract ----------------------
		mu_data = df_cat.loc[:,self.OBS].to_numpy()
		stds    = df_cat.loc[:,self.UNC].to_numpy()
		cr_data = df_cat.loc[:,self.RHO].to_numpy()
		#-------------------------------------

		#----- There is no correlation with r_vel ---
		idx_tru = np.triu_indices(6,k=1)
		idi     = np.where(idx_tru[1] != 5)[0]
		idx_tru = (idx_tru[0][idi],idx_tru[1][idi])
		#--------------------------------------------

		#-------- sd to diag ------------------
		sd_data = np.zeros((n_sources,6,6))
		diag = np.einsum('...jj->...j',sd_data)
		diag[:] = stds 
		#--------------------------------------

		one = np.eye(6)
		pbar = tqdm(total=n_sources,miniters=10000)
		for i,(sd,corr) in enumerate(zip(sd_data,cr_data)):
			#------- Correlation ---------
			rho  = np.zeros((6,6))
			rho[idx_tru] = corr
			rho  = rho + rho.T + one
			#-----------------------------

			#---------- Covariance ----------------------
			sg_data[i] = sd.dot(rho.dot(sd))
			#--------------------------------------------

			pbar.update()
		pbar.close()

		#----- Select members and field ----------
		ids_all  = df_cat["source_id"].to_numpy()
		ids_mem  = df_mem["source_id"].to_numpy()
		mask_mem = np.isin(ids_all,ids_mem)
		idx_cls  = np.where(mask_mem)[0]
		idx_fld  = np.where(~mask_mem)[0]
		#-----------------------------------------

		#-------------- Members -----------------------------
		assert len(idx_cls) > 1, "Error: Empty members file!"
		df_cat["Member"] = False
		df_cat.loc[mask_mem,"Member"] = True
		#----------------------------------------------------
		
		#---------- Random sample of field sources ------------------
		idx_rnd  = np.random.choice(idx_fld,size=n_fld,replace=False)
		mu_field = mu_data[idx_rnd]
		#------------------------------------------------------------
		
		#------------- Write -----------------------------------------
		df_cat.to_hdf(file_hdf,key="catalogue",mode="w")

		with h5py.File(file_data, 'w') as hf:
			hf.create_dataset('mu',         chunks=True,data=mu_data)
			hf.create_dataset('sg',         chunks=True,data=sg_data)
			hf.create_dataset('Cluster',    chunks=True,data=mu_syn)
			hf.create_dataset('Field',      chunks=True,data=mu_field)
			hf.create_dataset('idx_field',  chunks=True,data=idx_fld)
			hf.create_dataset('idx_cluster',chunks=True,data=idx_cls)
		#---------------------------------------------------------
		print("Data correctly assembled")

	def infer_models(self,list_components=range(1,10),case="Field",instance="Real"):

		file_data = self.file_data_base.format(instance)

		#------------ Read data------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			X = np.array(hf.get(case))
		#----------------------------------------

		#-- Dimensions --
		N,D = X.shape
		#----------------

		#-------------------- Loop over models ----------------------------------
		for n_components in list_components:
			file_model = self.file_model_base.format(instance,case,n_components)

			if os.path.isfile(file_model):
				continue

			#------------ Inference ---------------------------------------------
			print("Inferring model with {0} components.".format(n_components))
			gmm = GaussianMixture(dimension=D,n_components=n_components)
			gmm.setup(X,uncertainty=None)
			gmm.fit(random_state=2)
			#--------------------------------------------------------------------

			#------- Write --------------------------------
			with h5py.File(file_model,'w') as hf:
				hf.create_dataset('G',    data=n_components)
				hf.create_dataset('pros', data=gmm.weights_)
				hf.create_dataset('means',data=gmm.means_)
				hf.create_dataset('covs', data=gmm.covariances_)
				hf.create_dataset('aic',  data=gmm.aic)
				hf.create_dataset('bic',  data=gmm.bic)
				hf.create_dataset('nmn',  data=N*np.min(gmm.weights_))
			#------------------------------------------------
			print("------------------------------------------")


	def plot_criteria(self,list_components=range(1,10),case="Field",instance="Real"):

		file_comparison = self.file_comparison.format(instance,case)

		aics = []
		bics = []
		nmin = []

		#--------------- Read models ---------------------------
		for n_components in list_components:
			file_model = self.file_model_base.format(instance,case,n_components)
			#--------- Read -------------------------------------------
			with h5py.File(file_model, 'r') as hf:
				aics.append(np.array(hf.get('aic')))
				bics.append(np.array(hf.get('bic')))
				nmin.append(np.array(hf.get('nmn')))
			#-----------------------------------------------------------

		#---- Arrays --------
		aics = np.array(aics)
		bics = np.array(bics)
		nmin = np.array(nmin)
		#---------------------
		
		#-----------Plot BIC,AIC,NMIN ------------------------
		plt.figure(figsize=(8,6))
		axl = plt.gca()
		axl.plot(list_components,bics,label="BIC")
		axl.plot(list_components,aics,label="AIC")
		axl.set_xlabel('Components')
		axl.set_ylabel("Criterion")
		axl.set_xticks(list_components)
		axl.legend(loc="upper left")

		axr = axl.twinx()
		axr.plot(list_components,nmin,
				ls="--",color="black",label="$N_{min}$")
		axr.set_yscale("log")
		axr.set_ylabel("N_stars in lightest components")
		axr.legend(loc="upper right")

		plt.savefig(file_comparison,bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------


	def plot_model(self,n_components,case="Field",instance="Real"):
		file_data  = self.file_data_base.format(instance)
		file_model = self.file_model_base.format(instance,case,n_components)
		#--------- Read model -----------------------------
		with h5py.File(file_model, 'r') as hf:
			n_components = np.array(hf.get('G'))
			weights      = np.array(hf.get('pros'))
			means        = np.array(hf.get('means'))
			covariances  = np.array(hf.get('covs'))
		#--------------------------------------------------

		#------------ Read data --------------------
		with h5py.File(file_data, 'r') as hf:
			X = np.array(hf.get(case))
		#--------------------------------------

		#--------------------------------------------------------
		pdf = PdfPages(filename=file_model.replace(".h5",".pdf"))

		#------------ Colorbar ----------------------------
		norm_feat = Normalize(vmin=min(weights),vmax=max(weights)) 

		handles = [mlines.Line2D([], [], color="black", marker=None, 
					linestyle="-",label="Model"),
				   mlines.Line2D([], [], color="w", marker=".", 
					markerfacecolor="black",markersize=5,label="Data") ]

		sm = cm.ScalarMappable(norm=norm_feat, cmap=self.cmap_feat)
		#---------------------------------------------------------

		#------ Positions  -----------------------------------------
		fig, axs = plt.subplots(nrows=2,ncols=2,figsize=None)
		for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],self.idxs[:3]):
			mxgr = np.ix_(range(n_components),idx,idx)

			#--------- Sources --------------------------
			ax.scatter(x=X[:,idx[0]],y=X[:,idx[1]],
						marker=".",s=1,
						c="gray",zorder=0,
						label="Data")
			#-------------------------------------------

			#---------------- Inferred model ------------------------------
			for w,mu,sg in zip(weights,means[:,idx],covariances[mxgr]):
				width, height, angle = get_principal(sg)
				ell  = Ellipse(mu,width=width,height=height,angle=angle,
						clip_box=ax.bbox,edgecolor=sm.cmap(sm.norm(w)),
						ls="-",fill=False)
				ax.add_patch(ell)

			ax.scatter(means[:,idx[0]],means[:,idx[1]], 
						c=sm.cmap(sm.norm(weights)),marker='s',s=1)
			#--------------------------------------------------------------

			#------------- Titles --------------------
			ax.set_xlabel(self.labels_obs[idx[0]])
			ax.set_ylabel(self.labels_obs[idx[1]])
			ax.locator_params(tight=True, nbins=3)
			#----------------------------------------

		axs[0,0].axes.xaxis.set_visible(False)
		axs[0,1].axes.yaxis.set_visible(False)
		axs[1,1].axis("off")
		axs[1,1].legend(handles=handles,loc='center',
						bbox_to_anchor=(0.25, 0., 0.5, 0.5))
		fig.colorbar(sm,ax=axs[1,1],fraction=0.3,anchor=(0.0,0.0),
			shrink=0.75,extend="both",label='Weights')
		plt.subplots_adjust(left=None, bottom=None, right=None, 
						top=None, wspace=0.0, hspace=0.0)
		pdf.savefig(dpi=100,bbox_inches='tight')
		plt.close()

		#------ Velocities -----------------------------------------
		fig, axs = plt.subplots(nrows=2,ncols=2,figsize=None)

		for ax,idx in zip([axs[0,0],axs[0,1],axs[1,0]],self.idxs[3:]):
			mxgr = np.ix_(range(n_components),idx,idx)

			#--------- Sources --------------------------
			ax.scatter(x=X[:,idx[0]],y=X[:,idx[1]],
						marker=".",s=1,
						c="gray",zorder=0,
						label="Data")
			#-------------------------------------------

			#---------------- Inferred model -----------------------------------
			for w,mu,sg in zip(weights,means[:,idx],covariances[mxgr]):
				width, height, angle = get_principal(sg)
				ell  = Ellipse(mu,width=width,height=height,angle=angle,
						clip_box=ax.bbox,edgecolor=sm.cmap(sm.norm(w)),
						ls="-",fill=False)
				ax.add_patch(ell)

			ax.scatter(means[:,idx[0]],means[:,idx[1]], 
						c=sm.cmap(sm.norm(weights)),marker='s',s=1)
			#--------------------------------------------------------------

			#------------- Titles --------------------
			ax.set_xlabel(self.labels_obs[idx[0]])
			ax.set_ylabel(self.labels_obs[idx[1]])
			ax.locator_params(tight=True, nbins=3)
			#----------------------------------------

		axs[0,0].axes.xaxis.set_visible(False)
		axs[0,1].axes.yaxis.set_visible(False)
		axs[1,1].axis("off")
		axs[1,1].legend(handles=handles,loc='center',
						bbox_to_anchor=(0.25, 0., 0.5, 0.5))
		fig.colorbar(sm,ax=axs[1,1],fraction=0.3,anchor=(0.0,0.0),
			shrink=0.75,extend="both",label='Weights')
		plt.subplots_adjust(left=None, bottom=None, right=None,
						top=None, wspace=0.0, hspace=0.0)
		pdf.savefig(dpi=100,bbox_inches='tight')
		plt.close()
		pdf.close()
			
	def compute_probabilities(self,best,instance="Real"):

		file_data  = self.file_data_base.format(instance)
		file_hdf   = self.file_hdf_base.format(instance)

		#------- Read data-------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			mu = np.array(hf.get("mu"))
			sg = np.array(hf.get("sg"))
		#------------------------------------

		#-- Dimensions --
		N,_ = mu.shape
		#----------------

		self._initialize_mcmichael()

		#----- Likelihoods ----------------------------------
		print("Computing likelihoods ...")
		llk = {}
		for case in ["Field","Cluster"]:
			#--------------- File --------------------
			file_model = self.file_model_base.format(
				instance,case,best[case])
			#-----------------------------------------

			#------------ Read model -------------------
			print("Reading parameters ...")
			with h5py.File(file_model, 'r') as hf:
				n_components = np.array(hf.get("G"))
				weights      = np.array(hf.get("pros"))
				means        = np.array(hf.get("means"))
				covariances  = np.array(hf.get("covs"))
			#-------------------------------------------

			#------------ Inference ------------------------------------
			gmm = GaussianMixture(dimension=6,n_components=best[case])
			gmm.setup(mu,uncertainty=sg)
			gllks = gmm.log_likelihoods(weights,means,covariances)
			#-----------------------------------------------------------

			llk[case+"_llk"] = logsumexp(gllks,axis=1,keepdims=True)		

		#---- Creates dataframe --------
		df = pd.DataFrame(data=llk)
		#--------------------------

		#------- Probability --------------------------------------
		print("Computing probabilities ...")
		df["prob_cls"] = df.apply(lambda x: np.exp(x["Cluster_llk"])/\
									(np.exp(x["Field_llk"]) + 
						 			 np.exp(x["Cluster_llk"])),axis=1)
		#----------------------------------------------------------

		#---------- Replace ----------------------------
		print("Saving data ...")
		df.to_hdf(file_hdf,key="probabilities",mode="a")
		#-----------------------------------------------

	def find_probability_threshold(self,bins = 5,covariate = "g",metric = "MCC"):
		#=========== Evaluate Classifier =============================
		file_data = [file_cls_data.format(s) for s in random_seeds]
		clq = ClassifierQuality(file_data=file_data,
								variate=variate,
								covariate=covariate,
								true_class=true_class)
		clq.confusion_matrix(bins=bins,metric=metric)
		clq.plots(file_plot=file_qua_plot.format(covariate,metric))
		clq.save(file_tex=file_qua_tex.format(covariate,metric))
		#=============================================================

	def plot_members(self):
		#------------- Reading -------------------------------
		print("Reading catalogue and probabilities ...")
		df_cat = pd.read_hdf(file_hdf,key="catalogue")
		df_pro = pd.read_hdf(file_hdf,key="probabilities")
		#-----------------------------------------------------

		#----- Candidates --------------------------------
		print("Selecting candidates ...")
		mask_cnd = df_pro["prob_cls"] >= probability_threshold
		df_cnd = df_cat.loc[mask_cnd].copy()
		df_cnd = df_cnd.merge(df_pro,left_index=True,
									right_index=True)
		ids_cnd = df_cnd["source_id"].to_numpy()
		#-------------------------------------------------

		#----- Members ------------------------------
		df_mem = df_cat.loc[df_cat["Member"]].copy()
		df_mem = df_mem.merge(df_pro,left_index=True,
									right_index=True)
		ids_mem = df_mem["source_id"].to_numpy()
		#--------------------------------------------

		#-------- Summary----------------------------
		ids_common = np.intersect1d(ids_mem,ids_cnd)
		ids_reject = np.setdiff1d(ids_mem,ids_cnd)
		ids_new    = np.setdiff1d(ids_cnd,ids_mem)
		print("Common: {0}".format(len(ids_common)))
		print("Rejected: {0}".format(len(ids_reject)))
		print("New: {0}".format(len(ids_new)))
		#---------------------------------------------

		#----------- Color ---------------------------
		df_mem["g_rp"] = df_mem["g"] - df_mem["rp"]
		df_cnd["g_rp"] = df_cnd["g"] - df_cnd["rp"]
		#---------------------------------------------

		#--------------------------------------------------------
		pdf = PdfPages(filename=file_plt)
		for plot in plots:
			fig = plt.figure()
			ax  = plt.gca()
			#--------- Sources --------------------------
			ax.scatter(x=df_mem[plot[0]],y=df_mem[plot[1]],
						marker="s",s=5,
						c=df_mem["prob_cls"],
						vmin=0,vmax=1,
						cmap=cmap_prob,
						zorder=0,
						label="Members")
			scb = ax.scatter(x=df_cnd[plot[0]],y=df_cnd[plot[1]],
						marker=".",s=3,
						c=df_cnd["prob_cls"],
						vmin=0,vmax=1,
						cmap=cmap_prob,
						zorder=0,
						label="Candidates")
			#-------------------------------------------

			#------------- Titles --------------------
			ax.set_xlabel(plot[0])
			ax.set_ylabel(plot[1])
			ax.locator_params(tight=True, nbins=3)
			#----------------------------------------]

			ax.legend()
			fig.colorbar(scb,shrink=0.75,extend="both",label='Probability')
			pdf.savefig(dpi=100,bbox_inches='tight')
			plt.close()
		pdf.close()

	def generate_synthetic(self,seeds):
		#================= Generate syntehtic data ====================
		for seed in seeds:
			#------------ Create simulation directory ------
			os.makedirs(self.dir_simulations.format(seed),exist_ok=True)
			#------------------------------------------------

			#---------- Generate cluster ---------------------------
			ama = Amasijo(photometric_args=photometric_args,
						  kalkayotl_file=self.file_kalkayotl,
						  seed=seed)

			ama.generate_cluster(file_cls_data.format(seed),
								n_stars=n_members,
								m_factor=m_factor)

			ama.plot_cluster(file_plot=file_cls_plot.format(seed))
			#-------------------------------------------------------

			#-------- Read cluster and field ---------------
			df_cls = pd.read_csv(file_cls_data.format(seed))
			fld = Table.read(file_field, format='fits')
			df_fld = fld.to_pandas()
			#-----------------------------------------------

			#------ Class -----------
			df_cls[class_name] = True
			df_fld[class_name] = False
			#-------------------------

			#------ Concatenate ------------------------------
			df = pd.concat([df_cls,df_fld],ignore_index=True)
			#------------------------------------------------

			#------ Extract ----------------------
			mu_data = df.loc[:,OBS].to_numpy()
			sd_data = df.loc[:,UNC].to_numpy()
			#------------------------------------

			#----- Select members and field ----------
			idx_cls  = np.where(df[class_name])[0]
			idx_fld  = np.where(~df[class_name])[0]
			#-----------------------------------------

			# print("Writting catalogue ...")
			# df.to_hdf(file_cat_data.format(seed),
			# 			key="catalogue",mode="w")
			del df # Release memory

			#----------- Covariance matrices ------------------
			print("Filling covariance matrices ...")

			#-------- sd to diag ------------------
			zeros = np.zeros((len(sd_data),6,6))
			diag = np.einsum('...jj->...j',zeros)
			diag[:] = np.square(sd_data)
			sg_data = zeros.copy()
			del diag
			del zeros
			#--------------------------------------
			
			#---------- Random sample of field sources ------------------
			idx_rnd  = np.random.choice(idx_fld,size=n_field,replace=False)
			mu_field = mu_data[idx_rnd]
			#------------------------------------------------------------
			
			#------------- Write --------------------------------
			print("Writting data ...")
			with h5py.File(file_syn_data.format(seed), 'w') as hf:
				hf.create_dataset('mu', chunks=True,data=mu_data)
				hf.create_dataset('sg', chunks=True,data=sg_data)
				hf.create_dataset('mu_field',   chunks=True,data=mu_field)
				hf.create_dataset('idx_field',  chunks=True,data=idx_fld)
				hf.create_dataset('idx_cluster',chunks=True,data=idx_cls)
			#----------------------------------------------------
			print("Data correctly written")
			del mu_field
			del mu_data
			del sg_data
			del sd_data
		#=================================================================


if __name__ == "__main__":
	#----------------- Directories -----------
	dir_repos= "/home/jolivares/Repos/"
	dir_main = "/home/jolivares/Cumulos/ComaBer/Mecayotl/"
	#-----------------------------------------

	#----------- Files --------------------------------------------
	file_kalkayotl = dir_main + "Real/Data/Cluster_statistics.csv"
	file_catalogue = dir_main + "Real/Data/ComaBer.fits"
	file_members   = dir_main + "Real/Data/members_Furnkranz+2019.csv"
	file_syn_cat   = dir_main + "Real/Data/ComaBer_33deg_simulation-result.fits"
	#--------------------------------------------------------------

	#----- Miscellaneous ------------------------------------------------
	n_syn = 1000
	probability_threshold = 0.95
	seeds = range(10)
	photometric_args = {
		"log_age": np.log10(8.0e8),    
		"metallicity":0.012,
		"Av": 0.0,         
		"mass_limit":10.0, 
		"bands":["V","I","G","BP","RP"]
	}
	m_factor = 2
	#------------------------------------
	#---------------------------------------------------------------------
	

	mcy = Mecayotl(path_main=dir_main,
				   path_amasijo=dir_repos+"Amasijo/",
				   path_mcmichael=dir_repos+"McMichael/")

	#----------- Real data analysis ------------------------------------
	# mcy.generate_true_cluster(file_kalkayotl=file_kalkayotl)
	# mcy.assemble_data(file_catalogue=file_catalogue,file_members=file_members,
	# 				  instance="Real")
	# mcy.plot_criteria(list_components=range(10,16,1),case="Field",instance="Real")
	# mcy.plot_criteria(list_components=range(1,11,1),case="Cluster",instance="Real")
	# mcy.plot_model(n_components=14,case="Field",instance="Real")
	# mcy.plot_model(n_components=5,case="Cluster",instance="Real")
	mcy.compute_probabilities(best={"Field":14,"Cluster":5},instance="Real")

