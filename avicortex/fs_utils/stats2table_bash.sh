#!/bin/bash

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# export SUBJECTS_DIR before running this code.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
path=`dirname $0`
sleep 1
cd $path

echo "SUBJECTS_DIR: $SUBJECTS_DIR"
list="`ls $SUBJECTS_DIR`"
echo "LIST: $list"

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Segmentation Stats
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

asegstats2table --subjects $list --meas volume --skip --statsfile wmparc.stats --all-segs --tablefile wmparc_stats.txt
asegstats2table --subjects $list --meas volume --skip --tablefile aseg_stats.txt
asegstats2table --subjects $list --meas volume --skip --statsfile wmparc.stats --tablefile aseg_presurf_hypos_stats.txt
asegstats2table --subjects $list --meas volume --skip --statsfile brainvol.stats --tablefile brainvol_stats.txt

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Desikan Killiany Atlas - Cortical Parcellation Stats
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

aparcstats2table --subjects $list --hemi lh --meas volume --skip --parc aparc --tablefile aparc_volume_lh.txt
aparcstats2table --subjects $list --hemi lh --meas thickness --skip --parc aparc --tablefile aparc_thickness_lh.txt
aparcstats2table --subjects $list --hemi lh --meas area --skip --parc aparc --tablefile aparc_area_lh.txt
aparcstats2table --subjects $list --hemi lh --meas meancurv --skip --parc aparc --tablefile aparc_meancurv_lh.txt
aparcstats2table --subjects $list --hemi lh --meas gauscurv --skip --parc aparc --tablefile aparc_gauscurv_lh.txt
aparcstats2table --subjects $list --hemi lh --meas foldind --skip --parc aparc --tablefile aparc_foldind_lh.txt
aparcstats2table --subjects $list --hemi lh --meas curvind --skip --parc aparc --tablefile aparc_curvind_lh.txt

aparcstats2table --subjects $list --hemi rh --meas volume --skip --parc aparc --tablefile aparc_volume_rh.txt
aparcstats2table --subjects $list --hemi rh --meas thickness --skip --parc aparc --tablefile aparc_thickness_rh.txt
aparcstats2table --subjects $list --hemi rh --meas area --skip --parc aparc --tablefile aparc_area_rh.txt
aparcstats2table --subjects $list --hemi rh --meas meancurv --skip --parc aparc --tablefile aparc_meancurv_rh.txt
aparcstats2table --subjects $list --hemi rh --meas gauscurv --skip --parc aparc --tablefile aparc_gauscurv_rh.txt
aparcstats2table --subjects $list --hemi rh --meas foldind --skip --parc aparc --tablefile aparc_foldind_rh.txt
aparcstats2table --subjects $list --hemi rh --meas curvind --skip --parc aparc --tablefile aparc_curvind_rh.txt

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Destrieux Atlas - Cortical Parcellation Stats
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas volume --skip -t lh_a2009s_volume.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas thickness --skip -t lh_a2009s_thickness.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas area --skip -t lh_a2009s_area.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas meancurv --skip -t lh_a2009s_meancurv.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas gauscurv --skip -t lh_a2009s_gauscurv.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas foldind --skip -t lh_a2009s_foldind.txt
aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas curvind --skip -t lh_a2009s_curvind.txt

aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas volume --skip -t rh_a2009s_volume.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas thickness --skip -t rh_a2009s_thickness.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas area --skip -t rh_a2009s_area.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas meancurv --skip -t rh_a2009s_meancurv.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas gauscurv --skip -t rh_a2009s_gauscurv.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas foldind --skip -t rh_a2009s_foldind.txt
aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas curvind --skip -t rh_a2009s_curvind.txt

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# B Atlas - Cortical Parcellation Stats
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas volume --skip -t lh_BA_volume.txt
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas thickness --skip -t lh_BA_thickness.txt
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas area --skip -t lh_BA_area.txt
aparcstats2table --hemi lh --subjects $list --parc BA_exvivo --meas meancurv --skip -t lh_BA_meancurv.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas volume --skip -t rh_BA_volume.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas thickness --skip -t rh_BA_thickness.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas area --skip -t rh_BA_area.txt
aparcstats2table --hemi rh --subjects $list --parc BA_exvivo --meas meancurv --skip -t rh_BA_meancurv.txt
