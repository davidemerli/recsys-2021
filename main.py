if __name__ == '__main__':


    #from Recommenders.Hybrid.SimilarityMergedHybridRecommender import SimilarityMergedHybridRecommender
    #from src.GraphBased.P3alphaRecommender import P3alphaRecommender
    from Recommenders.GraphBased.RP3betaCBFRecommender import RP3betaCBFRecommender
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
    from Recommenders.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
    from Recommenders.Implicit.FeatureCombinedImplicitALSRecommender import FeatureCombinedImplicitALSRecommender
    from Recommenders.Hybrid.GeneralizedSimilarityMergedHybridRecommender import GeneralizedSimilarityMergedHybridRecommender
    from Utils.ICM_preprocessing import *
    from Utils.data_loader import load_URM, load_ICM
    from Utils.confidence_scaling import *
    from Utils.write_submission import write_submission

    URM_all = load_URM("input/data_train.csv")
    ICM_all = load_ICM("input/data_ICM_subgenre.csv")

    ICM_combined = combine(ICM=ICM_all, URM = URM_all)

    # p3alpha_recommender = P3alphaRecommender(
    #     URM_train=URM_all,
    #     verbose=False
    # )
    #
    # p3alpha_recommender.fit(
    #     topK=int(212.8832860130684),
    #     alpha=0.4729294763382114,
    #     implicit=True
    # )

    # RP3 BETA (Combined) -----------------------------------------

    rp3betaCombined_recommender= RP3betaCBFRecommender(
        URM_train=URM_all,
        ICM_train=ICM_combined,
        verbose=False
    )

    rp3betaCombined_recommender.fit(
        topK=int(529.1628484087545),
        alpha=0.45304737831676245,
        beta=0.226647894170121,
        implicit=False
    )

    # IALS -----------------------------------------

    IALS_recommender= FeatureCombinedImplicitALSRecommender(
        URM_train=URM_all,
        ICM_train=ICM_all,
        verbose=True
    )

    IALS_recommender.fit(
        factors=int(398.601583855084),
        regularization=0.01,
        use_gpu=False,
        iterations=int(94.22855449116447),
        num_threads=6,
        confidence_scaling=linear_scaling_confidence,
        **{
            'URM': {"alpha": 42.07374324671451},
            'ICM': {"alpha": 41.72067133975204}
        }
    )

    # SLIM -----------------------------------------    

    SLIM_recommender = MultiThreadSLIM_ElasticNet(
            URM_train=ICM_combined.T,
            verbose=False
        )

    SLIM_recommender.fit(
        alpha=0.00026894910579512645,
        l1_ratio=0.08074126876487486,
        topK=int(395.376118479588),
        workers=6
    )

    SLIM_recommender.URM_train = URM_all



    # OTHER STUFF ----------------------------------

    # rp3betaCBF_recommender= RP3betaCBFRecommender(
    #     URM_train=URM_all,
    #     ICM_train=ICM_all,
    #     verbose=False
    # )
    #
    # rp3betaCBF_recommender.fit(
    #     topK=int(117.1),
    #     alpha=0.9882,
    #     beta=0.7703,
    #     implicit=False
    # )

    # itemKNN_recommender= ItemKNNCFRecommender(
    #     URM_train=URM_all,
    #     verbose=False
    # )
    #
    # itemKNN_recommender.fit(
    #     topK=100,
    #     shrink=50
    # )
    #
    # pureSVD_recommender= PureSVDItemRecommender(
    #     URM_train=URM_all,
    #     verbose=False
    # )
    #
    #
    # pureSVD_recommender.fit(
    #     num_factors=772,
    #     topK= 599
    # )

    # Generalized Merged Hybrid -----------------------------------------

    recommender = GeneralizedMergedHybridRecommender(
        URM_train=URM_all,
        recommenders=[
            IALS_recommender,
            rp3betaCombined_recommender,
            SLIM_recommender
        ],
        verbose=False
    )

    recommender.fit(
        alphas=[
            0.6836750866517823,
            0.45969038157844144,
            0.2723405593515382
        ]
    )

    write_submission(recommender=recommender, target_users_path="input/data_target_users_test.csv",
                     out_path='output/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))