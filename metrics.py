import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

class ModelEvaluator:
    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted

    def roc_dataframe(self):
        # Create a dataframe of the fpr, tpr, and threshold values
        fpr, tpr, thresholds = roc_curve(self.actual, self.predicted)
        return pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    
    def precision_recall_dataframe(self):
        precision, recall, thresholds = precision_recall_curve(self.actual, self.predicted)
        #return precision[-5:],thresholds[-5:]
        return pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': list(thresholds) +[1]})

    def metric_summary(self, threshold):
        fpr, tpr, thresholds = roc_curve(self.actual, self.predicted)
        metric_dict = {}
        metric_dict['auc'] = auc(fpr, tpr)
        metric_dict['accuracy'] = accuracy_score(self.actual, self.predicted > threshold)
        metric_dict['precision'] = precision_score(self.actual, self.predicted > threshold)
        metric_dict['recall'] = recall_score(self.actual, self.predicted > threshold)
        metric_dict['f1_score'] = f1_score(self.actual, self.predicted > threshold)
        metric_dict['matthews_coeficient'] = matthews_corrcoef(self.actual, self.predicted > threshold)
        #metric_dict['confusion_matrix'] = confusion_matrix(self.actual, self.predicted > threshold)
        return metric_dict

    # Plot the ROC curve
    def plot_roc_curve(self,threshold=None, fill=True):
        auc_val = self.metric_summary(threshold)['auc']
        chart_title = f'ROC Curve (AUC={auc_val:.4f})'
        
        roc_line = alt.Chart(self.roc_dataframe()).mark_line(color = 'blue').encode(
                alt.X('fpr'),
                alt.Y('tpr',)
                ).interactive()

        if fill==True:
            roc = alt.Chart(self.roc_dataframe()).mark_area(fillOpacity = 0.15, fill = 'blue').encode(
                alt.X('fpr',),
                alt.Y('tpr',))

        diag_line = alt.Chart(self.roc_dataframe()).mark_line(strokeDash=[20,5], color = 'black').encode(
                                                                        alt.X('thresholds', scale = alt.Scale(domain=[0, 1]), title="false positive rate"),
                                                                        alt.Y('thresholds', scale = alt.Scale(domain=[0, 1]), title="true positive rate")).interactive()
        if threshold != None:
            roc_df = self.roc_dataframe()
            thresh_x = roc_df[roc_df['thresholds']>threshold].tail(1)['fpr'].values[0]
            threshold_line = alt.Chart(pd.DataFrame({'threshold': [thresh_x]})).mark_rule(color='red').encode(
                x='threshold:Q'
            )

        return ((roc_line + roc + threshold_line + diag_line).properties(
            title=alt.TitleParams(text=chart_title, fontSize=16), 
            width=400,
            height=400
            ).interactive()
            )
    
    def plot_tpr_fpr(self,threshold):
        # Create the Altair plot
        df_adj = self.roc_dataframe()[self.roc_dataframe()['thresholds']<=1.0]
        base = (alt.Chart(df_adj).mark_line().encode(
            x='thresholds',
            y='fpr',
            color=alt.value('purple'),
            tooltip=['thresholds', 'fpr']
        ).properties(
            title=alt.TitleParams(text='False Positive Rate and True Positive Rate vs Threshold', fontSize=16),
            width=400,
            height=400
        ) + alt.Chart(df_adj).mark_line().encode(
            #x='thresholds',
            #y='tpr',
            alt.X('thresholds',title='Threshold'),
            alt.Y('tpr',title='True Positive Rate False Positive Rate'),
            color=alt.value('blue'),
            tooltip=['thresholds', 'tpr']
        )
        )
        
        threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color='red').encode(
            x='threshold:Q'
        )

        return base + threshold_line

    
    def plot_precision_recall(self,threshold=None):
        # Create the precision-recall plot using Altair
        df = self.precision_recall_dataframe().sort_values('precision')
        precision_recall_plot = (alt.Chart(df).mark_line().encode(
            alt.X('recall:Q', title='Recall'),
            alt.Y('precision:Q', title='Precision'),
            tooltip=['precision','recall','thresholds']
        ).properties(
            title=alt.TitleParams(text='Precision vs. Recall', fontSize=16),
            width=400,
            height=400
        )
        )

        # Add a point to highlight the thresholds
        #threshold_points = alt.Chart(df).mark_circle(size=60, color='red').encode(
        #    alt.X('recall:Q'),
        #    alt.Y('precision:Q'),
        #    tooltip=['thresholds']
        #).interactive()

        return precision_recall_plot #+ threshold_points
    
    def confusion_matrix_dataframe(self,threshold):
        cm = confusion_matrix(self.actual, self.predicted > threshold)
        return pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    
    def plot_confusion_matrix(self,threshold):
        # Reset the index and convert the DataFrame to long format
        df = self.confusion_matrix_dataframe(threshold)
        df = df.reset_index().melt('index')
        #return df

        #Create the heatmap using Altair
        heatmap = alt.Chart(df).mark_rect().encode(
            x=alt.X('variable:N', title='Predicted'),
            y=alt.Y('index:N', title='Actual'),
            color='value:Q'
        )

        # Add text labels to the heatmap
        text = heatmap.mark_text(baseline='middle', fontSize=16).encode(
            text='value:Q',
            color=alt.condition(
                alt.datum.value >= df['value'].max() / 2,
                alt.value('white'),
                alt.value('black')
            )
        )

        # Combine the heatmap and text labels
        chart = (heatmap + text).properties(
            title='Confusion Matrix Heatmap',
            width=400,
            height=400
        ).interactive()

        # Remove axis labels and ticks
        chart = chart.configure_axis(
            labels=False,
            ticks=False
        )

        # Remove chart padding
        chart = chart.configure_view(
            strokeOpacity=0
        )
        return chart
    
    def actual_pred_dataframe(self):
        return pd.DataFrame({'Actual':self.actual,'Predicted': self.predicted})
    
    def plot_pred_densities(self,threshold):
        df = self.actual_pred_dataframe()
        chart = alt.Chart(df).transform_density(
            'Predicted',
            groupby=['Actual'],
            as_=['Predicted', 'density'],
        ).mark_area().encode(
            x="Predicted:Q",
            y='density:Q',
            color='Actual:O'
        ).properties(
            title='Probability Densities',
            width=900,
            height=500
        ).interactive()

        threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color='red').encode(
            x='threshold:Q'
        )
        return chart + threshold_line
            