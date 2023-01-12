from typing import Dict, List, Tuple, Optional, Callable

from type_infer.dtype import dtype
from lightwood.helpers.ts import filter_ds

from posthoc_xai.helpers import log
from posthoc_xai.base import BaseAnalysisBlock


def model_analyzer(
        predictor: Callable,
        data,  # : EncodedDs, TODO refactor to DF
        train_data,  # : EncodedDs, TODO refactor to DF
        stats_info,  # : StatisticalAnalysis, TODO refactor to dict with generic params
        target: str,
        tss,  # TimeseriesSettings, TODO: dict
        dtype_dict: Dict[str, str],
        accuracy_functions,
        ts_analysis: Dict,
        analysis_blocks: Optional[List[BaseAnalysisBlock]] = []
) -> Tuple[Dict, Dict[str, object]]:
    """
    Analyses model on a validation subset to evaluate accuracy, estimate feature importance and generate a
    calibration model to estimating confidence in future predictions.

    Additionally, any user-specified analysis blocks (see class `BaseAnalysisBlock`) are also called here.

    :return:
    runtime_analyzer: This dictionary object gets populated in a sequential fashion with data generated from
    any `.analyze()` block call. This dictionary object is stored in the predictor itself, and used when
    calling the `.explain()` method of all analysis blocks when generating predictions.

    model_analysis: `ModelAnalysis` object that contains core analysis metrics, not necessarily needed when predicting.
    """

    runtime_analyzer = {}
    data_type = dtype_dict[target]

    # retrieve encoded data representations
    encoded_train_data = train_data
    encoded_val_data = data
    data = encoded_val_data.data_frame
    input_cols = list([col for col in data.columns if col != target])

    # predictive task
    is_numerical = data_type in (dtype.integer, dtype.float, dtype.num_tsarray, dtype.quantity)
    is_classification = data_type in (dtype.categorical, dtype.binary, dtype.cat_tsarray)
    is_multi_ts = tss.is_timeseries and tss.horizon > 1

    # TODO: send from LW
    # has_pretrained_text_enc = any([isinstance(enc, PretrainedLangEncoder)
    #                                for enc in encoded_train_data.encoders.values()])

    # raw predictions for validation dataset
    args = {} if not is_classification else {"predict_proba": True}
    filtered_df = filter_ds(encoded_val_data, tss)
    # encoded_val_data = EncodedDs(encoded_val_data.encoders, filtered_df, encoded_val_data.target)
    encoded_val_data = filtered_df  # TODO: make this work

    # TODO: modify LW so that if args are dict, it changes into PA class internally
    normal_predictions = predictor(encoded_val_data, args=args)
    normal_predictions = normal_predictions.set_index(encoded_val_data.data_frame.index)

    # ------------------------- #
    # Run analysis blocks, both core and user-defined
    # ------------------------- #
    kwargs = {
        'predictor': predictor,
        'target': target,
        'input_cols': input_cols,
        'dtype_dict': dtype_dict,
        'normal_predictions': normal_predictions,
        'data': filtered_df,
        'train_data': train_data,
        'encoded_val_data': encoded_val_data,
        'is_classification': is_classification,
        'is_numerical': is_numerical,
        'is_multi_ts': is_multi_ts,
        'stats_info': stats_info,
        'tss': tss,
        'ts_analysis': ts_analysis,
        'accuracy_functions': accuracy_functions,
        'has_pretrained_text_enc': stats_info.get('has_pretrained_text_enc', False)
    }

    for block in analysis_blocks:
        log.info("The block %s is now running its analyze() method", block.__class__.__name__)
        runtime_analyzer = block.analyze(runtime_analyzer, **kwargs)

    # ------------------------- #
    # Populate ModelAnalysis object
    # ------------------------- #
    # TODO: build this ModelAnalysis in LW instead
    model_analysis = {
        'accuracies': runtime_analyzer.get('score_dict', {}),
        'accuracy_histogram': runtime_analyzer.get('acc_histogram', {}),
        'accuracy_samples': runtime_analyzer.get('acc_samples', {}),
        'train_sample_size': len(encoded_train_data),
        'test_sample_size': len(encoded_val_data),
        'confusion_matrix': runtime_analyzer.get('cm', []),
        'column_importances': runtime_analyzer.get('column_importances', {}),
        'histograms': stats_info.histograms,
        'dtypes': dtype_dict,
        'submodel_data': predictor.submodel_data  # TODO turn into LW-agnostic
    }

    return model_analysis, runtime_analyzer
