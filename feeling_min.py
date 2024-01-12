import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.platypus import Image
from sklearn.cluster import KMeans
import os

class SentimentAnalysisApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Analyse de Sentiment")
        
        self.label = tk.Label(self.master, text="Choisissez un fichier CSV:")
        self.label.pack(pady=10)
        
        self.button = tk.Button(self.master, text="Parcourir", command=self.load_csv)
        self.button.pack(pady=10)
        
        self.text = scrolledtext.ScrolledText(self.master, height=10, width=50)
        self.text.pack(pady=10)
        
        self.analyze_button = tk.Button(self.master, text="Analyser le sentiment", command=self.analyze_sentiment)
        self.analyze_button.pack(pady=10)
        

        self.result_text = scrolledtext.ScrolledText(self.master, height=5, width=50)
        self.result_text.pack(pady=10)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.df.name = os.path.basename(file_path)  # Utilisez le nom du fichier comme nom de base de données
            self.text.insert(tk.END, f"Fichier chargé: {file_path}\n")
            self.display_database_info()  # Affichez les informations de la base de données
            self.generate_pdf_report(0, "")  #générer le rapport PDF avec les informations de la base de données

    def analyze_sentiment(self):
        if hasattr(self, 'df'):
            self.df['Sentiment'] = self.df['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            self.text.insert(tk.END, "Analyse de sentiment terminée.\n")

            average_sentiment = self.df['Sentiment'].mean()
            sentiment_result = "Moyenne du sentiment: {:.2f}".format(average_sentiment)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, sentiment_result)
            # CREER LE DIAGRAMME
            opinions_chart_path = os.path.join("C:\\Users\\admin\\Desktop\\master2023_2024\\data_mining", "opinions_chart.png")
            self.figure_opinions, self.ax_opinions = plt.subplots(figsize=(5, 3), dpi=100)
            self.ax_opinions.hist(self.df['Sentiment'], bins=20, edgecolor='black')
            self.ax_opinions.set_title('Distribution des opinions')
            self.ax_opinions.set_xlabel('Opinion')
            self.ax_opinions.set_ylabel("Nombre d'avis")
            self.figure_opinions.savefig(opinions_chart_path, bbox_inches='tight')
            plt.close()
            self.text.insert(tk.END, f"Graphe des opinions généré: {opinions_chart_path}\n")
            # Texte analytique
            self.result_text.insert(tk.END, "\n\nAnalyse:\n")
            if average_sentiment > 0.2:
                self.result_text.insert(tk.END, "Les avis indiquent généralement un sentiment positif envers le produit/le service.\n")
            elif average_sentiment < -0.2:
                self.result_text.insert(tk.END, "Les avis indiquent généralement un sentiment négatif envers le produit/le service.\n")
                # Appel de la nouvelle méthode pour inclure les informations dans le rapport PDF
                pdf_path = os.path.join("C:\\Users\\admin\\Desktop\\master2023_2024\\data_mining", "rapport_sentiment.pdf")
                self.generate_pdf_report(average_sentiment, pdf_path)
            else:
                self.result_text.insert(tk.END, "Les avis sont plutôt neutres envers le produit.\n")

            # Générer le rapport PDF
            pdf_path = os.path.join("C:\\Users\\admin\\Desktop\\master2023_2024\\data_mining", "rapport_sentiment.pdf")
            self.generate_pdf_report(average_sentiment, pdf_path)

        else:
            self.text.insert(tk.END, "Veuillez charger un fichier CSV d'abord.\n")

    def display_database_info(self):
        if hasattr(self, 'df'):
            self.text.insert(tk.END, f"Nom de la base de données: {self.df.name}\n")
            self.text.insert(tk.END, f"Colonnes de la base de données: {', '.join(self.df.columns)}\n")
            self.text.insert(tk.END, f"Nombre de lignes dans la base de données: {len(self.df)}\n")
        else:
            self.text.insert(tk.END, "Aucune base de données chargée\n")

    def generate_pdf_report(self, average_sentiment, pdf_path):
        # Vérifier si le répertoire existe, sinon le créer
        directory = os.path.dirname(pdf_path)
        if not directory:
          print("Error: The directory path is not specified.")
        else:
            try:
        # Attempt to create the directory
               os.makedirs(directory)
               print(f"Directory '{directory}' created successfully.")
            except FileExistsError:
             print(f"Directory '{directory}' already exists.")
            except Exception as e:
             print(f"Error creating directory '{directory}': {e}")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Ajout de la moyenne du sentiment au rapport
        content = [Paragraph("Rapport d'analyse de sentiment", styles['Title']),
                   Paragraph("Moyenne du sentiment: {:.2f}".format(average_sentiment), styles['BodyText'])]

        # Ajout du texte analytique au rapport
        analysis_text = "Analyse:\n"
        # Ajout des informations de la base de données au rapport PDF
        content.append(Paragraph("Description de la base de données:", styles['BodyText']))
        if hasattr(self, 'df'):
         content.append(Paragraph(f"Nom de la base de données: {self.df.name}", styles['BodyText']))
         content.append(Paragraph(f"Colonnes de la base de données: {', '.join(self.df.columns)}", styles['BodyText']))
        else:
         content.append(Paragraph("Aucune base de données chargée", styles['BodyText']))
        content.append(Paragraph(f"Nombre de lignes dans la base de données: {len(self.df)}", styles['BodyText']))
        if average_sentiment > 0.2:
            analysis_text += "Les avis indiquent généralement un sentiment positif envers le produit."
        elif average_sentiment < -0.2:
            analysis_text += "Les avis indiquent généralement un sentiment négatif envers le produit."
        else:
            analysis_text += "Les avis sont plutôt neutres envers le produit."
        content.append(Paragraph(analysis_text, styles['BodyText']))

        # Ajout du graphe en cercle au rapport
        pie_chart_path = os.path.join(directory, "pie_chart.png")
        self.plot_pie_chart(pie_chart_path)
        content.append(Paragraph("Répartition des sentiments (Diagramme circulaire) :", styles['BodyText']))

        # Spécifier la taille pour le graphe en cercle
        pie_chart_width = 400
        pie_chart_height = 400

        # Ajouter le graphe en cercle avec la taille spécifiée
        content.append(Image(pie_chart_path, width=pie_chart_width, height=pie_chart_height))
        # Ajout du tableau au rapport
        table_data = [['Type d\'avis', 'Nombre'],
                      ['Positif', len(self.df[self.df['Sentiment'] > 0])],
                      ['Négatif', len(self.df[self.df['Sentiment'] < 0])],
                      ['Neutre', len(self.df[self.df['Sentiment'] == 0])]]
        table = Table(table_data, colWidths=[200, 200], style=TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                                                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                                                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                                                           ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                                                           ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                                                           ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                                                           ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        content.append(Paragraph("Nombre d'avis par type:", styles['BodyText']))
        content.append(table)
        # Ajout du graphe d'opinions au rapport
        opinions_chart_path = os.path.join(directory, "opinions_chart.png")
        self.plot_opinions(opinions_chart_path)
        content.append(Paragraph("Distribution des opinions:", styles['BodyText']))
        content.append(Image(opinions_chart_path, width=400, height=300))
        # Add k-means clustering information to the report
        # Ajouter les informations sur le clustering k-means au rapport
        # Ajouter les informations sur le clustering k-means au rapport
        if 'Sentiment' in self.df.columns:
            chemin_k_means_chart = os.path.join(directory, "k_means_chart.png")
            self.plot_k_means_chart(chemin_k_means_chart)
            content.append(Paragraph("Clustering K-Means (Opinions) :", styles['BodyText']))

            # Texte descriptif pour le graphique du clustering k-means
            description_k_means = (
                "Les résultats du clustering k-means représentent différents groupes d'opinions en fonction des scores de sentiment. "
                "Chaque point dans le graphique appartient à un cluster spécifique, et les clusters sont codés en couleur pour plus de clarté."
            )
            content.append(Paragraph(description_k_means, styles['BodyText']))

            # Ajouter le graphique du clustering k-means au PDF
            content.append(Image(chemin_k_means_chart, width=400, height=300))

            # Sauvegarder la description du graphique du clustering k-means dans un fichier texte
            chemin_description_k_means = os.path.join(directory, "description_k_means.txt")
            with open(chemin_description_k_means, 'w') as fichier:
                fichier.write(description_k_means)

            # Ajouter le fichier de description du graphique du clustering k-means au contenu du PDF
            texte_description_k_means = (
                "Pour une explication détaillée du graphique du clustering k-means, veuillez vous référer au fichier texte : "
                f"{chemin_description_k_means}"
            )
            content.append(Paragraph(texte_description_k_means, styles['BodyText']))

            # Ajouter un tableau avec des informations sur les clusters au PDF
            tableau_info_cluster = self.get_cluster_info_table()
            content.append(tableau_info_cluster)
        doc.build(content)
        self.text.insert(tk.END, f"Rapport PDF généré: {pdf_path}\n")
    def get_cluster_info_table(self):
        if 'Sentiment' in self.df.columns:
            X = self.df[['Sentiment']]
            kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
            self.df['Cluster'] = kmeans.labels_

            # Compter le nombre de points de données dans chaque cluster
            compte_clusters = self.df['Cluster'].value_counts().sort_index()

            # Créer un tableau avec des informations sur les clusters
            donnees_tableau = [['Cluster', 'Nombre de points de données']]
            for cluster, compte in compte_clusters.items():
                donnees_tableau.append([cluster, compte])

            tableau = Table(donnees_tableau, colWidths=[100, 150], style=TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black)]))

            return tableau
        else:
            print("Erreur : la colonne 'Sentiment' n'a pas été trouvée dans le DataFrame.")
            return None

    def plot_k_means_chart(self, k_means_chart_path=None):
        if k_means_chart_path is None:
            k_means_chart_path = os.path.join("C:\\Users\\admin\\Desktop\\master2023_2024\\data_mining", "k_means_chart.png")

        # Perform k-means clustering on 'Sentiment' column
        if 'Sentiment' in self.df.columns:
            X = self.df[['Sentiment']]
            kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
            self.df['Cluster'] = kmeans.labels_

            # Plot k-means clustering results
            fig, ax = plt.subplots()
            scatter = ax.scatter(X['Sentiment'], [1] * len(X), c=self.df['Cluster'], cmap='viridis')
            ax.set_title('K-Means Clustering Results')
            ax.set_xlabel('Sentiment')
            ax.set_yticks([])
            ax.legend(*scatter.legend_elements(), title="Clusters")
            fig.savefig(k_means_chart_path, bbox_inches='tight')
            plt.close()
        else:
            print("Error: 'Sentiment' column not found in the DataFrame.")


    def plot_pie_chart(self, pie_chart_path):
     if 'Sentiment' in self.df.columns:
        positive_reviews = len(self.df[self.df['Sentiment'] > 0])
        negative_reviews = len(self.df[self.df['Sentiment'] < 0])
        neutral_reviews = len(self.df[self.df['Sentiment'] == 0])

        labels = ['Positif', 'Négatif', 'Neutre']
        sizes = [positive_reviews, negative_reviews, neutral_reviews]
        colors = ['#66b3ff', '#ff8899', '#99ff99']

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.4))

        # Déplacer les étiquettes à l'extérieur du cercle
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontalalignment, **kw)

        ax.axis('equal')  # Assure que le cercle est dessiné correctement

        # Sauvegarder le graphe en cercle
        plt.savefig(pie_chart_path, bbox_inches='tight')
        plt.close()
     else:
        print("Error: 'Sentiment' column not found in the DataFrame.")
    def plot_opinions(self, opinions_chart_path=None):
        if opinions_chart_path is None:
        # Si aucun chemin n'est spécifié, utilisez un chemin par défaut
          opinions_chart_path = os.path.join("C:\\Users\\admin\\Desktop\\master2023_2024\\data_mining", "opinions_chart.png")

        self.figure_opinions, self.ax_opinions = plt.subplots(figsize=(5, 3), dpi=100)
        self.ax_opinions.hist(self.df['Sentiment'], bins=20, edgecolor='black')
        self.ax_opinions.set_title('Distribution des opinions')
        self.ax_opinions.set_xlabel('Opinion')
        self.ax_opinions.set_ylabel("Nombre d'avis")
        self.figure_opinions.savefig(opinions_chart_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()

