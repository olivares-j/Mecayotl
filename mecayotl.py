import sys
import os
import numpy as np
import pandas as pd
import h5py
import dill

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
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

	def __init__(self,path_main,
					nc_cluster=range(1,10),
					nc_field=range(1,10),
					path_mcmichael = "/home/jolivares/Repos/McMichael/",
					path_amasijo   = "/home/jolivares/Repos/Amasijo/",
					cmap_probability="viridis_r",
					cmap_features="viridis_r",
					seed=1234):

		gaia_observables = [
		"ra","dec","parallax","pmra","pmdec","dr2_radial_velocity",
		"ra_error","dec_error","parallax_error","pmra_error","pmdec_error","dr2_radial_velocity_error",
		"ra_dec_corr","ra_parallax_corr","ra_pmra_corr","ra_pmdec_corr",
		"dec_parallax_corr","dec_pmra_corr","dec_pmdec_corr",
		"parallax_pmra_corr","parallax_pmdec_corr",
		"pmra_pmdec_corr"]

		#------ Set Seed -----------------
		np.random.seed(seed=seed)
		self.random_state = np.random.RandomState(seed=seed)
		#----------------------------------------------------

		#------------- Directories -----------------
		self.dir_mcmi       = path_mcmichael
		self.dir_amasijo    = path_amasijo
		self.path_main      = path_main
		#--------------------------------------

		#---------------- Files --------------------------------------------------
		self.file_smp_base   = path_main + "/{0}/Data/members_synthetic.csv"
		self.file_hdf_base   = path_main + "/{0}/Data/catalogue.h5"
		self.file_data_base  = path_main + "/{0}/Data/data.h5"
		self.file_model_base = path_main + "/{0}/Models/{1}_GMM_{2}.h5"
		self.file_comparison = path_main + "/{0}/Models/{1}_comparison.png"
		self.file_qlt_base   = path_main + "/Classification/quality_{0}_{1}.{2}"
		self.file_mem_data   = path_main + "/Classification/members_mecayotl.csv"
		self.file_mem_plot   = path_main + "/Classification/members_mecayotl.pdf"
		#-------------------------------------------------------------------------

		#-------------- Parameters -----------------------------------------------------
		self.cmap_prob = plt.get_cmap(cmap_probability)
		self.cmap_feat = plt.get_cmap(cmap_features)
		self.idxs      = [[0,1],[2,1],[0,2],[3,4],[5,4],[3,5]]
		self.plots     = [["ra","dec"],["pmra","pmdec"],["parallax","pmdec"],["g_rp","g"]]
		self.OBS       = gaia_observables[0:6]
		self.UNC       = gaia_observables[6:12]
		self.RHO       = gaia_observables[12:]
		self.nc_case   = {"Field":nc_field,"Cluster":nc_cluster}
		self.best_gmm  = {}
		#----------------------------------------------------------------------------------

	def _initialize_mcmichael(self):
		#-------------- Commands to replace dimension -----------------------------
		cmd = 'sed -e "s|DIMENSION|{0}|g"'.format(6)
		cmd += ' {0}GPU/Functions_base.py > {0}GPU/Functions.py'.format(self.dir_mcmi)
		os.system(cmd)
		#--------------------------------------------------------------------------
		sys.path.append(self.dir_mcmi)

	def _initialize_amasijo(self):
		sys.path.append(self.dir_amasijo)

	def generate_true_cluster(self,file_kalkayotl,n_samples=100000,instance="Real"):
		"""
		Generate synthetic data based on Kalkayotl input parameters
		"""
		file_smp  = self.file_smp_base.format(instance)

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
		df.to_csv(file_smp,index_label="source_id")
		#---------------------------------------------------


	def assemble_data(self,file_catalogue,file_members,
						n_fld=100000,instance="Real"):
		#------------ Files ------------------------------
		file_data = self.file_data_base.format(instance)
		file_hdf  = self.file_hdf_base.format(instance)
		file_smp  = self.file_smp_base.format(instance)
		#-------------------------------------------------

		#--------------- Read ------------------------------
		cat = Table.read(file_catalogue, format='fits')
		df_cat = cat.to_pandas()
		df_mem = pd.read_csv(file_members,usecols=["source_id"])
		df_syn = pd.read_csv(file_smp,usecols=self.OBS)
		#---------------------------------------------------

		n_sources = df_cat.shape[0]
		mu_syn    = df_syn.to_numpy()
		sg_syn    = np.zeros((len(mu_syn),6,6))

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
		sg_field = sg_data[idx_rnd]
		#------------------------------------------------------------
		
		#------------- Write -----------------------------------------
		df_cat.to_hdf(file_hdf,key="catalogue",mode="w")

		with h5py.File(file_data, 'w') as hf:
			hf.create_dataset('mu',         chunks=True,data=mu_data)
			hf.create_dataset('sg',         chunks=True,data=sg_data)
			hf.create_dataset('mu_Cluster', chunks=True,data=mu_syn)
			hf.create_dataset('mu_Field',   chunks=True,data=mu_field)
			hf.create_dataset('sg_Cluster', chunks=True,data=sg_syn)
			hf.create_dataset('sg_Field',   chunks=True,data=sg_field)
			hf.create_dataset('idx_Field',  chunks=True,data=idx_fld)
			hf.create_dataset('idx_Cluster',chunks=True,data=idx_cls)
		#---------------------------------------------------------
		print("Data correctly assembled")

	def infer_models(self,case="Field",instance="Real"):

		#---------- Libraries ------------------
		self._initialize_mcmichael()
		from GPU.gmm import GaussianMixture
		#---------------------------------------

		file_data = self.file_data_base.format(instance)

		#------------ Read data------------------
		print("Reading data ...")
		with h5py.File(file_data, 'r') as hf:
			X = np.array(hf.get("mu_"+case))
			U = np.array(hf.get("sg_"+case))
		#----------------------------------------

		#-- Dimensions --
		N,_ = X.shape
		#----------------

		#-------------------- Loop over models ----------------------------------
		for n_components in self.nc_case[case]:
			file_model = self.file_model_base.format(instance,case,n_components)

			if os.path.isfile(file_model):
				continue

			#------------ Inference ---------------------------------------------
			print("Inferring model with {0} components.".format(n_components))
			gmm = GaussianMixture(dimension=6,n_components=n_components)
			gmm.setup(X,uncertainty=U)
			gmm.fit(random_state=self.random_state)
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


	def select_best_model(self,case="Field",instance="Real",
							minimum_nmin=100):

		file_comparison = self.file_comparison.format(instance,case)

		aics = []
		bics = []
		nmin = []

		#--------------- Read models ---------------------------
		for n_components in self.nc_case[case]:
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

		#------ Find best ---------------------------------
		idx_valid = np.where(nmin > minimum_nmin)[0]
		idx_min   = np.argmin(aics[idx_valid])
		idx_best  = idx_valid[idx_min]
		#--------------------------------------------------

		#------------ Set best model ------------------------------------------
		if instance in self.best_gmm.keys():
			self.best_gmm[instance].update({case:self.nc_case[case][idx_best]})
		else:
			self.best_gmm.update({instance:{case:self.nc_case[case][idx_best]}})
		#-----------------------------------------------------------------------
		
		#-----------Plot BIC,AIC,NMIN ------------------------
		plt.figure(figsize=(8,6))
		axl = plt.gca()
		axl.plot(self.nc_case[case],bics,label="BIC")
		axl.plot(self.nc_case[case],aics,label="AIC")
		axl.set_xlabel('Components')
		axl.set_ylabel("Criterion")
		axl.set_xticks(self.nc_case[case])
		axl.legend(loc="upper left")

		axr = axl.twinx()
		axr.plot(self.nc_case[case],nmin,
				ls="--",color="black",label="$N_{min}$")
		axr.set_yscale("log")
		axr.set_ylabel("N_stars in lightest components")
		axr.legend(loc="upper right")

		axr.axvline(x=self.nc_case[case][idx_best],
						linewidth=1, color='gray',ls=":")

		plt.savefig(file_comparison,bbox_inches='tight')
		plt.close()
		#-----------------------------------------------------


	def plot_model(self,n_components=None,case="Field",instance="Real"):
		#--------------- n_components ----------------------------------------
		if n_components is None:
			n_components = self.best_gmm[instance][case]
		assert isinstance(n_components,int), "n_components must be an integer!"
		#----------------------------------------------------------------------

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
			X = np.array(hf.get("mu_"+case))
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
			ax.set_xlabel(self.OBS[idx[0]])
			ax.set_ylabel(self.OBS[idx[1]])
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
			ax.set_xlabel(self.OBS[idx[0]])
			ax.set_ylabel(self.OBS[idx[1]])
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
			
	def compute_probabilities(self,instance="Real"):

		#---------- Libraries ------------------
		self._initialize_mcmichael()
		from GPU.gmm import GaussianMixture
		#---------------------------------------

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
			n_components = self.best_gmm[instance][case]
			#--------------- File --------------------------
			file_model = self.file_model_base.format(
				instance,case,n_components)
			#------------------------------------------------

			#------------ Read model -------------------
			print("Reading {0} parameters ...".format(case))
			with h5py.File(file_model, 'r') as hf:
				n_components = np.array(hf.get("G"))
				weights      = np.array(hf.get("pros"))
				means        = np.array(hf.get("means"))
				covariances  = np.array(hf.get("covs"))
			#-------------------------------------------

			#------------ Inference ------------------------------------
			gmm = GaussianMixture(dimension=6,n_components=n_components)
			gmm.setup(mu,uncertainty=sg)
			gllks = gmm.log_likelihoods(weights,means,covariances)
			#-----------------------------------------------------------

			#---------- Total likelihood -----------------------------
			llk[case+"_llk"] = logsumexp(gllks,axis=1,keepdims=False)
			#---------------------------------------------------------

		#------- Probability --------------------------------------
		print("Computing probabilities ...")
		llk["prob_cls"] = np.exp(llk["Cluster_llk"])/\
			(np.exp(llk["Field_llk"]) + np.exp(llk["Cluster_llk"]))
		#----------------------------------------------------------

		#---------- Append probability -------------------
		df_cat = pd.read_hdf(file_hdf,key="catalogue",mode="r")
		df_cat["prob_cls"] = llk["prob_cls"]
		#-----------------------------------------------

		print("Updating catalogue ...")
		df_cat.to_hdf(file_hdf,key="catalogue",format="table",mode="w")
		del df_cat

	def generate_synthetic(self,file_field,file_kalkayotl,
							photometric_args,
							n_members=1000,
							n_field=100000,
							seeds=range(1),m_factor=2):

		#-------- Libraries --------
		self._initialize_amasijo()
		from Amasijo import Amasijo
		#---------------------------

		#================= Generate syntehtic data ====================
		for seed in seeds:
			#--------- Directory ---------------------------------
			name_base = "/Synthetic_{0}/".format(seed)
			dir_sim   = self.path_main + name_base
			file_smp  = self.file_smp_base.format(name_base)
			file_data = self.file_data_base.format(name_base)
			file_hdf  = self.file_hdf_base.format(name_base)
			#----------------------------------------------------

			#--- Create simulation directory -----------
			os.makedirs(dir_sim,exist_ok=True)
			os.makedirs(dir_sim + "Data",exist_ok=True)
			os.makedirs(dir_sim + "Models",exist_ok=True)
			#-------------------------------------------

			# #---------- Generate cluster ---------------------------
			ama = Amasijo(photometric_args=photometric_args,
						  kalkayotl_file=file_kalkayotl,
						  seed=seed)

			ama.generate_cluster(file_smp,n_stars=n_members,
								m_factor=m_factor)

			ama.plot_cluster(file_plot=file_smp.replace(".csv",".pdf"))
			#----------------------------------------------------------

			#-------- Read cluster and field ---------------
			print("Reading field and cluster data ...")
			df_cls = pd.read_csv(file_smp)
			fld    = Table.read(file_field, format='fits')
			df_fld = fld.to_pandas().sample(n=n_field,
										random_state=seed)
			#-----------------------------------------------

			#------ Rename field radial_velocity -----------------
			df_fld.rename(columns={"radial_velocity":self.OBS[-1],
						   "radial_velocity_error":self.UNC[-1]},
						   inplace=True)
			#-----------------------------------------------------

			#------ Class -----------
			df_cls["Cluster"] = True
			df_fld["Cluster"] = False
			#-------------------------

			#------ Concatenate ------------------------------
			df = pd.concat([df_cls,df_fld],ignore_index=True)
			#------------------------------------------------

			#------ Extract -----------------------
			mu_data = df.loc[:,self.OBS].to_numpy()
			sd_data = df.loc[:,self.UNC].to_numpy()
			#--------------------------------------

			#----- Select members and field -------
			idx_cls  = np.where( df["Cluster"])[0]
			idx_fld  = np.where(~df["Cluster"])[0]
			#--------------------------------------

			#-------- Write & delete ------------------------
			df.to_hdf(file_hdf,key="catalogue",mode="w")
			del df # Release memory
			#----------------------------------------------

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

			#--------------- Write data ------------------------------------
			with h5py.File(file_data, 'w') as hf:
				hf.create_dataset('mu',         chunks=True,data=mu_data)
				hf.create_dataset('sg',         chunks=True,data=sg_data)
				hf.create_dataset('mu_Cluster', chunks=True,data=mu_data[idx_cls])
				hf.create_dataset('mu_Field',   chunks=True,data=mu_data[idx_fld])
				hf.create_dataset('sg_Cluster', chunks=True,data=sg_data[idx_cls])
				hf.create_dataset('sg_Field',   chunks=True,data=sg_data[idx_fld])
				hf.create_dataset('idx_Field',  chunks=True,data=idx_fld)
				hf.create_dataset('idx_Cluster',chunks=True,data=idx_cls)
			#----------------------------------------------------------------
			print("Data correctly written")
			del mu_data
			del sg_data
			del sd_data

	def compute_probabilities_synthetic(self,seeds):

		for i,seed in enumerate(seeds):
			#------------ File and direcotry ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			dir_model = "{0}/{1}/Models/".format(self.path_main,instance)
			file_hdf  = self.file_hdf_base.format(instance)
			os.makedirs(dir_model,exist_ok=True)
			#-------------------------------------------------------------
			print("Analyzing {0} data ...".format(instance))

			if i == 0:
				#---------- Infer field model --------------------------
				self.infer_models(case="Field",instance=instance)
				self.select_best_model(case="Field",instance=instance)
				self.plot_model(case="Field",instance=instance)
				#-------------------------------------------------------

				#-------- Best cluster same as Real data ---------------
				self.best_gmm[instance].update({"Cluster":
									self.best_gmm["Real"]["Cluster"]})
				#-------------------------------------------------------

			else:
				#---------------- Copy field models ------------------------
				model_fld = self.file_model_base.format(
					"Synthetic_{0}".format(seeds[0]),"Field",
					self.best_gmm["Synthetic_{0}".format(seeds[0])]["Field"])
				cmd_fld   = "cp {0} {1}".format(model_fld,dir_model)
				os.system(cmd_fld)
				#-----------------------------------------------------

				#-------- Best gmm same as seed_0 ---------------
				self.best_gmm[instance] = self.best_gmm[
								"Synthetic_{0}".format(seeds[0])]
				#-------------------------------------------------------
			#---------------------------------------------------------------

			#------------ Cluster model ------------------------------
			model_cls = self.file_model_base.format("Real","Cluster",
							self.best_gmm["Real"]["Cluster"])
			cmd_cls   = "cp {0} {1}".format(model_cls,dir_model)
			os.system(cmd_cls)
			#---------------------------------------------------------

			#----- Compute probabilities -----------------
			self.compute_probabilities(instance=instance)
			#---------------------------------------------

	def find_probability_threshold(self,seeds,bins=4,covariate="g",metric="ACC"):
		#-------- Libraries -------------------
		self._initialize_amasijo()
		from Quality import ClassifierQuality
		#--------------------------------------

		os.makedirs(self.path_main+"/Classification/")

		dfs = []
		for seed in seeds:
			#------------ File ----------------------------
			instance  = "Synthetic_{0}".format(seed)
			file_hdf  = self.file_hdf_base.format(instance)
			
			#------------- Reading -------------------------------
			print("Reading catalogue and probabilities ...")
			df = pd.read_hdf(file_hdf,key="catalogue",
					columns=["Cluster","prob_cls",covariate])
			#-----------------------------------------------------

			#-- Append ----
			dfs.append(df)
			#--------------

		print("Analyzing classifier quality ...")
		clq = ClassifierQuality(file_data=dfs,
								variate="prob_cls",
								covariate=covariate,
								true_class="Cluster")
		clq.confusion_matrix(bins=bins,metric=metric,prob_steps=100)
		clq.plots(file_plot=self.file_qlt_base.format(covariate,metric,"pdf"))
		clq.save(file_tex=self.file_qlt_base.format(covariate,metric,"tex"))

		self.file_thresholds = self.file_qlt_base.format(
											covariate,metric,"pkl")

	def plot_members(self,probability_threshold=None,instance="Real"):

		file_hdf  = self.file_hdf_base.format(instance)

		#------------- Reading -------------------------------
		print("Reading catalogue ...")
		df_cat = pd.read_hdf(file_hdf,key="catalogue")
		#-----------------------------------------------------

		#----- Members ------------------------------
		df_mem  = df_cat.loc[df_cat["Member"]].copy()
		#--------------------------------------------

		#----- Candidates --------------------------------
		print("Selecting candidates ...")
		if isinstance(probability_threshold,float):
			mask_cnd = df_cat["prob_cls"] >= probability_threshold
			df_cnd   = df_cat.loc[mask_cnd].copy()
			del df_cat

		else:
			#------------- File tresholds -----------------------------------
			file_thresholds = self.file_thresholds if probability_threshold \
						is None else probability_threshold
			#----------------------------------------------------------------

			#----- Load edges and probability thresholds ------
			with open(file_thresholds,'rb') as in_stream: 
				quality = dill.load(in_stream)
			#--------------------------------------------------

			#------- Split data frame into bins ---------------------
			bin_mag = np.digitize(df_cat[quality["covariate"]].values,
						bins=quality["edges"])
			#--------------------------------------------------------

			#------ Bin 0 objects to bin 1----------------
			# Few objects are brighter than the brightest edge so 
			# we use the same probability as the first bin
			bin_mag[np.where(bin_mag == 0)[0]] = 1
			#-------------------------------------

			#----------- Loop over bins ----------------------------
			dfs = []
			for i,threshold in enumerate(quality["thresholds"]):
				#--------- Objects in bin -----------------------
				idx = np.where(bin_mag == i)[0]
				strategy = "Bin {0}".format(i)
				# There are no objects in bin zero
				# so we use it for all objects
				if i == 0: 
					assert len(idx) == 0 ,"Bin 0 is not empty!"
					idx = np.arange(len(df_cat))
					strategy = "All"
				#------------------------------------------------

				#----------- Temporal DF ---------
				tmp = df_cat.iloc[idx].copy()
				#----------------------------------

				#------------- Members ----------------------------
				tmp_mem = tmp[tmp["prob_cls"] >= threshold].copy()
				tmp_mem["Strategy"] = strategy
				#-------------------------------------------------

				#------------ Append -------------------------------------------
				dfs.append(tmp_mem)
				#---------------------------------------------------------------

			#--------- Concatenate and extract -----
			df = pd.concat(dfs)
			df_cnd = df[df["Strategy"] != "All"].copy()
			del df_cat
			#-------------------------------------------
		#-------------------------------------------------

		#---------- IDs ------------------------
		ids_cnd = df_cnd["source_id"].to_numpy()
		ids_mem = df_mem["source_id"].to_numpy()
		#---------------------------------------

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
		pdf = PdfPages(filename=self.file_mem_plot)
		for i,plot in enumerate(self.plots):
			fig = plt.figure()
			ax  = plt.gca()
			#--------- Sources --------------------------
			ax.scatter(x=df_mem[plot[0]],y=df_mem[plot[1]],
						marker=".",s=5,
						c=df_mem["prob_cls"],
						vmin=0,vmax=1,
						cmap=self.cmap_prob,
						zorder=0,
						label="Members")
			scb = ax.scatter(x=df_cnd[plot[0]],y=df_cnd[plot[1]],
						marker="$\u25A1$",
						s=10,c=df_cnd["prob_cls"],
						vmin=0,vmax=1,
						cmap=self.cmap_prob,
						zorder=0,
						label="Candidates")
			#-------------------------------------------

			#------------- Titles --------------------
			ax.set_xlabel(plot[0])
			ax.set_ylabel(plot[1])
			ax.locator_params(tight=True, nbins=3)
			#----------------------------------------

			#------- Invert ----------
			if i==3:
				ax.invert_yaxis()
			#-----------------------

			ax.legend()
			fig.colorbar(scb,shrink=0.75,extend="both",label='Probability')
			pdf.savefig(dpi=100,bbox_inches='tight')
			plt.close()
		pdf.close()
		#-------------------------------------------------------------------

		#--------- Save candidates -------
		df_cnd.to_csv(self.file_mem_data,index=False)
		#=============================================================


if __name__ == "__main__":
	#----------------- Directories ------------------------
	dir_repos= "/scratch/jolivares/Repos/"
	dir_main = "/scratch/jolivares/OCs/ComaBer/Mecayotl/"
	dir_cats = "/scratch/jolivares/OCs/ComaBer/Catalogues/"
	#-------------------------------------------------------

	#----------- Files --------------------------------------------
	file_kalkayotl = dir_main + "Real/Data/Cluster_statistics.csv"
	file_members   = dir_main + "Real/Data/members_Furnkranz+2019.csv"
	file_new_mem   = dir_main + "Real/Data/members_mecayotl.csv"
	file_rel_cat   = dir_cats + "ComaBer_33deg.fits"
	file_syn_cat   = dir_cats + "ComaBer_33deg_simulation.fits"
	#--------------------------------------------------------------

	#----- Miscellaneous ------------------------------------------------
	seeds = range(10)
	photometric_args = {
		"log_age": np.log10(8.0e8),    
		"metallicity":0.012,
		"Av": 0.0,         
		"mass_limit":10.0, 
		"bands":["V","I","G","BP","RP"]
	}
	bins = [4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
	#---------------------------------------------------------------------
	

	mcy = Mecayotl(nc_cluster=range(1,11,1),
				   nc_field=range(1,15,1),
				   path_main=dir_main,
				   path_amasijo=dir_repos+"Amasijo/",
				   path_mcmichael=dir_repos+"McMichael/")

	#----------- Real data analysis ------------------------------------------
	# mcy.generate_true_cluster(file_kalkayotl=file_kalkayotl)
	# mcy.assemble_data(file_catalogue=file_rel_cat,file_members=file_members,
	# 														instance="Real")
	# mcy.infer_models(case="Field",instance="Real")
	# mcy.infer_models(case="Cluster",instance="Real")
	# mcy.select_best_model(case="Field",instance="Real")
	# mcy.select_best_model(case="Cluster",instance="Real")
	# mcy.plot_model(case="Field",instance="Real")
	# mcy.plot_model(case="Cluster",instance="Real")
	# mcy.best_gmm = {'Real': {'Field': 11, 'Cluster': 10}}
	# mcy.compute_probabilities(instance="Real")
	#-------------------------------------------------------------------------

	#----------- Synthetic data --------------------------------
	mcy.generate_synthetic(file_field=file_syn_cat,
						   file_kalkayotl=file_kalkayotl,
						   photometric_args=photometric_args,
						   n_field=int(1e6),
						   seeds=seeds)
	mcy.compute_probabilities_synthetic(seeds)
	#----------------------------------------------------------
	
	mcy.find_probability_threshold(seeds=seeds,bins=bins)
	mcy.plot_members(instance="Real")
	#----------------------------------------------------------------------------

	


