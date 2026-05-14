from django.db import models


class Document(models.Model):
	file = models.FileField(upload_to='documents/')
	uploaded_at = models.DateTimeField(auto_now_add=True)
	# Add any other fields you need (e.g., title, type, etc.)

	def __str__(self):
		return self.file.name


class Module(models.Model):
	document = models.ForeignKey('Document', related_name='modules', on_delete=models.CASCADE)
	name = models.CharField(max_length=512)
	order = models.PositiveIntegerField(default=0)

	class Meta:
		ordering = ['order']

	def __str__(self):
		return self.name


class Subsection(models.Model):
	module = models.ForeignKey('Module', related_name='subsections', on_delete=models.CASCADE)
	name = models.CharField(max_length=512)
	content = models.TextField()
	order = models.PositiveIntegerField(default=0)

	class Meta:
		ordering = ['order']

	def __str__(self):
		return self.name
